"""Central configuration for Unprepared.

Every tunable that used to be a magic number inside ``main.py`` lives here so
that balancing, resolution changes or new content never require touching the
simulation or rendering code.  All values default to the exact numbers used by
the original prototype so behaviour is unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "textures")

Color3 = Tuple[float, float, float]
Color4 = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class WindowSettings:
    width: int = 1024
    height: int = 768
    title: str = "Unprepared"
    near_plane: float = 0.1
    far_plane: float = 100.0
    target_fps: int = 60
    font_name: str = "Arial"
    font_size: int = 20
    font_small_size: int = 16
    tag_font_name: str = "Courier New"   # bold monospace used for the (S) stolen tag


@dataclass(frozen=True)
class PlayerSettings:
    walk_speed: float = 4.0            # world units / second
    sprint_multiplier: float = 1.8
    crouch_speed_multiplier: float = 0.5   # sneaking is slow
    attention_turn_speed: float = 150.0    # degrees/s the view swings round when an NPC demands attention
    mouse_sensitivity: float = 0.15    # degrees per pixel
    max_pitch: float = 89.0
    # ``player.position`` is at the feet (y = the floor), like an NPC's. The townsfolk stand 1.74 tall
    # with their eyes at 1.6, so the camera sits a touch below that: nobody towers over anybody.
    eye_height: float = 1.55           # camera offset above ``player.position`` when standing
    crouch_eye_height: float = 0.9     # ... and when crouching
    stance_transition_speed: float = 4.0   # eye height change per second between the two
    # Collision: the player is a box ``collision_radius`` wide around ``position`` reaching from
    # ``position.y + collision_foot_clearance`` (low lips and steps do not catch the feet) up to
    # ``position.y + collision_head_height``.
    collision_radius: float = 0.3
    collision_foot_clearance: float = 0.25
    collision_head_height: float = 1.7
    # Stealth model
    noise_walk: float = 1.5
    noise_sprint: float = 3.0
    noise_crouch_multiplier: float = 0.3
    noise_floor: float = 0.1
    clothing_noise_floor: float = 0.25     # the quietest an outfit can make the player's steps (x1 = bare feet)
    visibility_torch_bonus: float = 0.9
    visibility_crouch_multiplier: float = 0.4


@dataclass(frozen=True)
class TimeSettings:
    start_hour: int = 7
    time_scale: float = 2.5            # simulated minutes per real second
    minutes_per_hour: float = 60.0


@dataclass(frozen=True)
class InteractionSettings:
    door_range: float = 5.0
    npc_range: float = 6.0             # talk / trade distance (eye to NPC feet)
    container_range: float = 4.0
    aim_dot_threshold: float = 0.6     # cos(angle) needed between view dir and target
    door_aim_offset: Tuple[float, float] = (1.5, 1.5)   # (x, y) offset from hinge to aim point
    door_facing_min: float = 0.25      # view dir component towards the door plane needed (0 = edge-on allowed)
    crosshair_ray: bool = True         # whatever the crosshair ray hits first is the target (before the aim cone)
    npc_aim_height: float = 1.0
    pickpocket_range: float = 2.5      # must sneak this close (feet to feet) to pick pockets


@dataclass(frozen=True)
class LockpickSettings:
    rotate_step_degrees: float = 10.0
    success_tolerance_degrees: float = 15.0
    default_sweet_spot: float = 45.0   # used when ``random_sweet_spot`` is False
    random_sweet_spot: bool = True     # new random sweet spot for every lock attempt
    reset_pick_each_attempt: bool = True
    consume_pick_on_snap: bool = True  # a snapped pick is removed from the inventory


@dataclass(frozen=True)
class NPCSettings:
    move_speed: float = 3.6            # world units / second while travelling (a brisk walk; the player sprints at 7.2)
    patrol_pause: float = 2.0          # seconds spent at each patrol waypoint
    arrival_tolerance: float = 0.15    # distance at which a waypoint counts as reached
    door_operate_reach: float = 0.9    # how close (beyond the door box) an NPC opens a door it walks through
    door_lock_clearance: float = 1.5   # distance past a door before an NPC locks it behind itself
    locked_out_retry: float = 15.0     # seconds between attempts to get into a place that was shut


@dataclass(frozen=True)
class NavigationSettings:
    cell_size: float = 0.5
    agent_radius: float = 0.4          # obstacles are inflated by this much
    bounds: Tuple[float, float, float, float] = (-25.0, -25.0, 25.0, 25.0)   # min_x, min_z, max_x, max_z


@dataclass(frozen=True)
class EconomySettings:
    """Three numbers per item (see ``TradePolicy``): its *worth* (``Item.value``, what is always
    shown), the merchant's *asking price* (worth plus a markup the Barter skill haggles down,
    never below the worth) and the merchant's *offer* for it (a fraction of the worth that grows
    a little with every Barter level and never exceeds the worth). Buying and selling the same
    thing back therefore always loses money."""
    stolen_goods_multiplier: float = 0.5    # a fence pays this much of the usual offer for hot goods
    purse_grab: int = 20               # coins lifted per successful pickpocket of a purse at Pickpocket level 1 ...
    purse_grab_per_level: int = 2      # ... plus this many per level above it
    merchant_markup: float = 0.25           # asking price = worth x (1 + markup) at Barter level 1 ...
    barter_markup_cut_per_level: float = 0.02   # ... and each level haggles 2% of the markup away (gone at level 51)
    sell_fraction_base: float = 0.25        # a merchant offers a quarter of the worth at Barter level 1 ...
    sell_fraction_per_level: float = 0.01   # ... one more percent of it per level ...
    sell_fraction_cap: float = 1.0          # ... and never more than the worth itself
    pantry_restock_days: int = 3            # households top their kitchen tables back up this often (midnight)


@dataclass(frozen=True)
class SkillRewards:
    """Base XP for each deed at an even match (challenge = base level); ``progression.grant``
    doubles it for every ``xp_doubling_levels`` the deed's challenge stands above the base level."""
    pickpocket: float = 20.0           # plus value-based XP for the item lifted
    smithing: float = 40.0
    smithing_fail: float = 15.0
    lockpicking: float = 45.0          # a lock picked (its challenge does the scaling)
    lockpicking_fail: float = 10.0
    stealing: float = 10.0             # plus value-based XP
    barter: float = 5.0                # plus value-based XP
    mining: float = 10.0
    sneak_success: float = 12.0        # a sneak roll passed in front of a watching NPC
    sneak_fail: float = 4.0
    secure_success: float = 60.0       # working a mounted, warded or chained treasure free
    secure_fail: float = 12.0


@dataclass(frozen=True)
class ProgressionSettings:
    """The single rule every skill follows (see ``systems/progression.py``):

    * effective level = base level (1 .. ``max_level``) + bonuses of worn gear
    * chance = ``floor + (ceiling - floor) * sigmoid((effective - challenge) / logistic_scale)``
    * XP = base * magnitude * 2 ** ((challenge - base level) / ``xp_doubling_levels``)
    """
    max_level: int = 100
    logistic_scale: float = 8.0        # levels of margin per e-fold in the odds
    chance_floor: float = 0.005
    chance_ceiling: float = 0.995
    xp_doubling_levels: float = 12.0
    xp_factor_min: float = 0.02
    xp_factor_max: float = 6.0
    value_xp_rate: float = 0.5         # XP magnitude per gold of item value
    value_xp_cap: float = 250.0
    value_xp_floor: float = 0.5        # junk still teaches a sliver
    noise_floor: float = 0.3           # the quietest Sneak makes your steps (x1 at level 1) ...
    noise_falloff: float = 30.0        # ... approached over about this many levels


@dataclass(frozen=True)
class PerceptionSettings:
    guard_view_distance: float = 14.0
    civilian_view_distance: float = 8.0
    torch_view_multiplier: float = 1.5
    min_light_view_multiplier: float = 0.4
    hearing_radius_per_noise: float = 10.0
    wake_noise_threshold: float = 0.8
    wall_noise_attenuation: float = 0.25   # noise that gets through walls (player and NPC in different buildings/outside)
    shout_radius: float = 60.0             # how far a cry of "Guards!" carries in the open ...
    shout_wall_attenuation: float = 0.5    # ... and the fraction of that through walls
    scold_cooldown: float = 15.0           # seconds before an NPC repeats the same warning
    scold_freeze_seconds: float = 1.2      # a guard's warning holds the player's attention (controls lock, view turns)
    warning_grace: float = 6.0             # seconds after a warning during which suspicion cannot reach the alarm
    torch_warnings_before_crime: int = 3   # "light a torch" warnings a guard gives before it becomes a crime
    racket_noise: float = 3.6              # a step this loud (as heard) is a racket awake people remark on
    # Suspicion rates are per second of simulated time (alarm at 100).
    seen_suspicion_gain: float = 80.0      # scaled by visibility / distance
    min_visibility_to_notice: float = 0.15
    suspicion_decay: float = 25.0
    trespass_suspicion_gain: float = 45.0  # extra per second while the player is somewhere they should not be
    torch_light_reach: float = 4.5         # an NPC's torch lights the player up within this distance ...
    torch_lit_visibility: float = 0.5      # ... to at least this visibility
    schedule_override_threshold: float = 50.0
    alarm_threshold: float = 100.0
    view_half_angle_degrees: float = 90.0   # NPCs see a half-space in front of them, never behind


# Lock tiers: what a lock is (0 = a villager's latch .. 5 = legendary) and how hard it is to pick.
LOCK_TIER_NAMES: Tuple[str, ...] = ("Flimsy", "Simple", "Sturdy", "Strong", "Master", "Legendary")
# Item rarities, in order.
RARITIES: Tuple[str, ...] = ("common", "uncommon", "rare", "epic", "legendary")


@dataclass(frozen=True)
class StealthSettings:
    """Every challenge rating in the game, on the skill scale (see ``ProgressionSettings``).

    Who is watching (``vigilance`` by role), how lit and how showy the player is, how rare the
    thing taken is and how good the lock is all add up to one number ``C``; the check is
    ``chance(effective level, C)``."""
    sneak_check_interval: float = 1.0      # seconds between sneak rolls per watching NPC
    sneak_fail_suspicion: float = 35.0
    # vigilance: an observer's (and a pocket-owner's) alertness, by role
    vigilance: Tuple[Tuple[str, float], ...] = (
        ("villager", -5.0), ("townsfolk", 0.0), ("merchant", 5.0), ("fence", 10.0), ("guard", 15.0),
        ("rich", 18.0), ("hunter", 12.0), ("bandit", 22.0), ("knight", 34.0), ("boss", 50.0),
    )
    sneak_visibility_challenge: float = 15.0    # added per unit of the player's visibility (light, torch, stance, clothes)
    asleep_challenge: float = -20.0             # a sleeper's pockets and senses
    suspicious_challenge: float = 10.0          # an observer already on edge (suspicion past the override threshold)
    bystander_challenge: float = 0.0            # bystanders noticing a dip into somebody's pockets
    theft_challenge: float = 5.0                # lifting something off a watched shelf unnoticed
    rarity_challenge: Tuple[Tuple[str, float], ...] = (
        ("common", 0.0), ("uncommon", 6.0), ("rare", 15.0), ("epic", 28.0), ("legendary", 45.0))
    # epic and legendary pieces are mounted, chained or warded: working one free is a Stealing
    # check of its own, and one this far beyond the player cannot even be attempted
    secure_challenge: Tuple[Tuple[str, float], ...] = (("epic", 28.0), ("legendary", 48.0))
    secure_gate: float = 30.0                   # levels below the challenge at which an attempt is hopeless
    lock_challenge: Tuple[float, ...] = (-25.0, -10.0, 5.0, 18.0, 32.0, 47.0)    # by lock tier (LOCK_TIER_NAMES)
    forge_challenge: float = 0.0
    mining_challenge: float = -10.0
    barter_challenge: float = 0.0
    # the lock mini-game, from the margin m = (E - C) / logistic_scale:
    lockpick_tolerance_base: float = 6.0        # degrees either side of the sweet spot at an even match ...
    lockpick_tolerance_per_margin: float = 3.0  # ... widening with margin ...
    lockpick_min_tolerance: float = 2.5         # ... within these bounds
    lockpick_max_tolerance: float = 24.0
    lockpick_wobble_base: float = 14.0          # degrees the hand wanders at an even match ...
    lockpick_wobble_per_margin: float = 4.0     # ... steadier with margin
    lockpick_max_wobble: float = 32.0
    # Sneak XP per second while crouch-moving: a trickle even alone, more near people who do
    # not notice you (sleepers and backs turned count as "behind"), most right under their noses;
    # each scaled by that person's vigilance as a challenge
    sneak_xp_alone: float = 0.75
    sneak_xp_behind: float = 3.0
    sneak_xp_in_front: float = 7.0
    sneak_xp_range: float = 14.0
    pickpocket_fail_xp_multiplier: float = 0.3
    stealing_fail_xp_multiplier: float = 0.4


@dataclass(frozen=True)
class ClothingSettings:
    """How what the player *wears* changes how easily they are seen and heard (see
    ``entities/equipment.py``). Every worn item contributes by its material (metal glints and
    clanks, leather is dull and fairly quiet, cloth is dim and silent), its shade (lighter than
    ``shade_pivot`` stands out, darker hides) and the slot it is worn in (the body shows most;
    shoes are what is heard). The sums become multipliers on the player's visibility and
    footstep noise, clamped to the floors/ceilings here."""
    visibility_metal: float = 0.35
    visibility_leather: float = -0.04
    visibility_cloth: float = -0.08
    noise_metal: float = 0.25
    noise_leather: float = -0.02
    noise_cloth: float = -0.08
    shade_pivot: float = 0.45              # luminance of a colour that neither helps nor hurts
    shade_slope: float = 0.5               # visibility per unit of luminance above/below the pivot
    visibility_weights: Tuple[Tuple[str, float], ...] = (("head", 0.5), ("neck", 0.9), ("body", 1.0), ("feet", 0.3), ("weapon", 0.25))
    noise_weights: Tuple[Tuple[str, float], ...] = (("head", 0.3), ("neck", 0.6), ("body", 1.0), ("feet", 2.0), ("weapon", 0.6))
    visibility_floor: float = 0.25         # even head-to-toe black cloth shows a little by day ...
    visibility_ceiling: float = 2.5        # ... and polished steel only glints so much
    noise_ceiling: float = 2.2
    still_crouch_multiplier: float = 0.5   # crouched *and* not moving: harder still to make out


@dataclass(frozen=True)
class Penalty:
    """What a crime costs: the fine, the days in the cells served *on top of* paying it, and
    the days served *instead* when the fine cannot be paid."""
    fine: float
    jail_days: int = 0
    jail_days_unpaid: int = 1


@dataclass(frozen=True)
class JusticeSettings:
    """Crimes and punishments (see README "Crimes and punishments").

    *Theft crimes* - pickpocketing, lockpicking, stealing from shelves and tables - are
    punished by where they were committed: ``outdoors`` or ``indoors`` (inside any building).
    Every other crime has a penalty of its own in ``penalties`` (``default_penalty`` for one
    that is not listed). On top of that, every stolen item the player is carrying when charged
    adds its value to the fine and ``stolen_item_jail_days`` to the sentence; one the player
    cannot pay for adds ``stolen_item_unpaid_jail_days`` more.
    """
    theft_crimes: Tuple[str, ...] = ("Pickpocketing", "Lockpicking a door", "Lockpicking a lockbox", "Shoplifting", "Theft")
    outdoors: Penalty = Penalty(100.0, jail_days=2, jail_days_unpaid=3)
    indoors: Penalty = Penalty(500.0, jail_days=5, jail_days_unpaid=7)
    penalties: Tuple[Tuple[str, Penalty], ...] = (
        ("Skulking about", Penalty(20.0, 0, 1)),         # ignoring a guard's "stand up straight" more than once
        ("No torch after dark", Penalty(5.0, 0, 1)),     # ignoring the torch law after repeated warnings
        ("Trespassing", Penalty(15.0, 0, 1)),
        ("Fencing stolen goods", Penalty(30.0, 0, 1)),   # offering stolen goods to an honest merchant
        ("Jailbreak", Penalty(100.0, 2, 3)),
    )
    default_penalty: Penalty = Penalty(20.0, 0, 1)
    stolen_item_jail_days: int = 1         # per stolen item carried when charged (its value joins the fine)
    stolen_item_unpaid_jail_days: int = 1  # ... and one more for every stolen item whose value cannot be paid
    release_hour: int = 6                # prisoners are let out at this hour of their last day
    charge_delay: float = 3.0            # seconds after a crime is declared before a guard may charge the player:
    #                                      the "Halt!" and the chase come first, so there is time to run
    charge_reach: float = 2.5            # a guard must get this close to lay hands on the player
    caught_freeze_seconds: float = 2.0   # controls lock while a guard has the player bang to rights
    jail_fade_delay: float = 2.0         # ... and the fade to the cell waits this long so it sinks in
    release_grace: float = 180.0         # seconds a released prisoner may linger to collect their things
    curfew_ask_interval: float = 8.0     # seconds between a resident's requests to leave at night
    curfew_asks_before_eviction: int = 3


@dataclass(frozen=True)
class CraftingSettings:
    lockpicks_min: int = 1                 # yield at level 1 ...
    lockpicks_max: int = 15                # ... and at ``lockpicks_max_level`` and beyond
    lockpicks_max_level: int = 50


@dataclass(frozen=True)
class LightingSettings:
    """Atmosphere and light colours (all linear RGB, intensities ~0..1.5)."""
    sky_day: Color3 = (0.46, 0.66, 0.95)
    sky_dusk: Color3 = (0.85, 0.45, 0.25)
    sky_night: Color3 = (0.02, 0.035, 0.09)     # natural dark blue
    fog_tint: Color3 = (0.7, 0.72, 0.75)
    fog_density_day: float = 0.006       # exp2 fog: barely a haze by day ...
    fog_density_night: float = 0.055     # ... thick darkness at night (about 15 units of visibility)
    sun_color_noon: Color3 = (1.05, 1.0, 0.92)
    sun_color_horizon: Color3 = (1.0, 0.55, 0.3)
    moon_color: Color3 = (0.16, 0.18, 0.26)
    ambient_day: Color3 = (0.38, 0.40, 0.45)
    ambient_night: Color3 = (0.05, 0.055, 0.08)
    torch_color: Color3 = (1.3, 0.9, 0.45)
    lantern_color: Color3 = (1.5, 1.1, 0.62)     # indoor wall lanterns
    candle_color: Color3 = (1.1, 0.8, 0.42)      # table candles
    fire_color: Color3 = (1.6, 0.9, 0.4)         # the smith's forge
    torch_range: float = 11.0
    player_torch_range: float = 14.0
    torch_flicker: float = 0.12
    underground_light: float = 0.12      # the share of the sky's light that reaches down into a cave
    glow_color: Color3 = (0.35, 0.95, 0.75)   # the cold light of cave mushrooms
    sun_disc_size: float = 9.0
    moon_disc_size: float = 6.0
    sky_distance: float = 85.0           # billboard distance for the sun/moon (inside the far plane)
    # fixed-function fallback (no GLSL)
    npc_torch_height: float = 1.2
    ambient_day_level: float = 1.0       # perception ambient light (0..1)
    ambient_night_level: float = 0.1


@dataclass(frozen=True)
class HudColors:
    panel: Color4 = (0.08, 0.08, 0.12, 0.85)
    top_bar: Color4 = (0.08, 0.08, 0.12, 0.9)
    prompt_panel: Color4 = (0.08, 0.08, 0.12, 0.95)
    modal_panel: Color4 = (0.05, 0.05, 0.08, 1.0)      # opaque: nothing drawn underneath shows through
    lockpick_panel: Color4 = (0.05, 0.05, 0.08, 1.0)
    white: RGB = (255, 255, 255)
    gold: RGB = (255, 215, 0)
    torch_on: RGB = (255, 200, 100)
    crouch_on: RGB = (150, 255, 150)
    danger: RGB = (255, 120, 120)
    prompt: RGB = (255, 240, 100)
    prompt_title: RGB = (235, 225, 200)                # the name over a prompt ("Guard House")
    log_title: RGB = (150, 220, 255)
    trade_heading: RGB = (100, 220, 255)
    selected: RGB = (255, 230, 100)
    sweet_spot_text: RGB = (255, 230, 0)
    stolen_tag: RGB = (255, 70, 70)
    scroll_hint: RGB = (170, 170, 190)
    bubble: Color4 = (0.85, 0.82, 0.7, 0.95)          # speech bubble border
    bubble_fill: Color4 = (0.12, 0.11, 0.14, 0.92)    # speech bubble panel
    bubble_name: RGB = (255, 215, 0)
    bubble_text: RGB = (245, 240, 225)
    # item names by rarity (common ones stay white)
    rarity_uncommon: RGB = (130, 235, 130)
    rarity_rare: RGB = (110, 175, 255)
    rarity_epic: RGB = (205, 125, 255)
    rarity_legendary: RGB = (255, 170, 60)


@dataclass(frozen=True)
class HudLayout:
    """Pixel layout of the HUD, derived from the window size.

    The defaults reproduce the original 1024x768 layout exactly; other window
    sizes keep panels anchored to the edges/centre they were designed around.
    """
    width: int = 1024
    height: int = 768
    top_bar_height: int = 45
    controls_panel: Tuple[int, int, int, int] = field(default=(720, 55, 1014, 275))
    inventory_panel: Tuple[int, int, int, int] = field(default=(10, 580, 350, 750))
    log_panel: Tuple[int, int, int, int] = field(default=(370, 560, 1014, 750))
    trade_panel: Tuple[int, int, int, int] = field(default=(212, 134, 812, 634))
    lockpick_panel: Tuple[int, int, int, int] = field(default=(312, 184, 712, 584))
    equipment_panel: Tuple[int, int, int, int] = field(default=(10, 55, 1014, 750))   # the [Tab] view fills the window
    prompt_y: int = 420
    lockpick_center: Tuple[int, int] = (512, 430)
    lockpick_radius: float = 90.0
    max_inventory_rows: int = 6
    max_log_rows: int = 6              # wrapped lines shown in the dialogue panel
    log_line_height: int = 22

    @classmethod
    def for_window(cls, width: int, height: int) -> "HudLayout":
        dx = width - 1024
        dy = height - 768
        return cls(
            width=width,
            height=height,
            controls_panel=(720 + dx, 55, 1014 + dx, 275),
            inventory_panel=(10, 580 + dy, 350, 750 + dy),
            log_panel=(370, 560 + dy, 1014 + dx, 750 + dy),
            trade_panel=(212 + dx // 2, 134 + dy // 2, 812 + dx // 2, 634 + dy // 2),
            lockpick_panel=(312 + dx // 2, 184 + dy // 2, 712 + dx // 2, 584 + dy // 2),
            equipment_panel=(10, 55, 1014 + dx, 750 + dy),
            prompt_y=420 + dy // 2,
            lockpick_center=(512 + dx // 2, 430 + dy // 2),
        )

    @property
    def center(self) -> Tuple[int, int]:
        return self.width // 2, self.height // 2


@dataclass(frozen=True)
class GameSettings:
    window: WindowSettings = WindowSettings()
    player: PlayerSettings = PlayerSettings()
    time: TimeSettings = TimeSettings()
    interaction: InteractionSettings = InteractionSettings()
    lockpick: LockpickSettings = LockpickSettings()
    economy: EconomySettings = EconomySettings()
    skill_rewards: SkillRewards = SkillRewards()
    progression: ProgressionSettings = ProgressionSettings()
    perception: PerceptionSettings = PerceptionSettings()
    npc: NPCSettings = NPCSettings()
    navigation: NavigationSettings = NavigationSettings()
    stealth: StealthSettings = StealthSettings()
    clothing: ClothingSettings = ClothingSettings()
    crafting: CraftingSettings = CraftingSettings()
    justice: JusticeSettings = JusticeSettings()
    lighting: LightingSettings = LightingSettings()
    hud_colors: HudColors = HudColors()
    asset_dir: str = DEFAULT_ASSET_DIR
    save_file: Optional[str] = None     # quick-save path; None = ``paths.default_save_file()`` (the user data directory)
    echo_messages_to_stdout: bool = True
    max_log_messages: int = 6
    random_seed: Optional[int] = None  # seed for gameplay randomness (lock sweet spots); None = unpredictable

    def hud_layout(self) -> HudLayout:
        return HudLayout.for_window(self.window.width, self.window.height)

    def resolved_save_file(self, override: Optional[str] = None) -> str:
        """The quick-save path to use: ``override``, else ``save_file``, else the per-user
        data directory (``paths.default_save_file``)."""
        from .paths import default_save_file
        return override or self.save_file or default_save_file()


DEFAULT_SETTINGS = GameSettings()
