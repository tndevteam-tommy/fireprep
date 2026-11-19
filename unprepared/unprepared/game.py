"""Headless game simulation.

``Game`` owns all mutable state (world, player, UI mode, mini-game and
transfer sessions) and exposes an *action API* (``interact()``,
``toggle_torch()``, ``transfer_confirm()`` ...) plus a per-frame ``update()``.
It imports neither pygame nor OpenGL, so the complete gameplay can be driven
and verified from plain unit tests; the ``input`` package merely translates
pygame events into these calls and the ``rendering`` package only reads from it.
"""
from __future__ import annotations

import random
import time
from enum import Enum
from typing import List, Optional, Tuple

from dataclasses import dataclass
from typing import Callable

from .config import DEFAULT_SETTINGS, GameSettings
from .core.messages import MessageLog, stdout_echo
from .entities import Player
from .systems import (ContainerPolicy, InteractionTarget, LockpickMinigame, LockpickOutcome, MoveIntent,
                      MovementResult, NPCAction, PickpocketPolicy, TradePolicy, TransferResult, TransferSession,
                      apply_movement, find_interaction_target, lockpick_tolerance, lockpick_wobble, noise_multiplier,
                      use_station)
from .entities import AIState
from .quests import QuestLog
from .tutorial import Tutorial
from .world import LevelDefinition, SaveManager, World, get_level
from .world.world import Relocation


class GameMode(Enum):
    EXPLORE = "explore"
    TRADE = "trade"            # transfer session with a merchant
    PICKPOCKET = "pickpocket"  # transfer session with a victim's pockets
    CONTAINER = "container"    # transfer session with a shelf / chest
    LOCKPICK = "lockpick"
    CONFIRM = "confirm"        # a yes/no question is on screen
    EQUIPMENT = "equipment"    # the [Tab] view: the player's figure and what they wear


@dataclass
class Confirmation:
    """A question the HUD shows until the player answers (Enter = yes, Esc = no, or a third key)."""
    text: str
    on_yes: Callable[[], None]
    on_no: Optional[Callable[[], None]] = None
    yes_label: str = "[ENTER] Yes"
    no_label: str = "[ESC] No"
    alt_label: Optional[str] = None          # text for the third option, if any
    on_alt: Optional[Callable[[], None]] = None


class ScreenFade:
    """A fade to black and back around a relocation (thrown out of a house, into a cell).

    ``alpha`` is the blackness the HUD draws (0 clear .. 1 black). The pending move is
    applied the moment the screen is fully black, so the player sees where they were,
    the world goes dark, and it comes back where they now are. Input is frozen meanwhile.
    """
    OUT_SECONDS = 0.6
    HOLD_SECONDS = 0.2
    IN_SECONDS = 0.7

    def __init__(self, on_black: Callable[[], None], delay: float = 0.0, out_seconds: Optional[float] = None,
                 hold_seconds: Optional[float] = None, in_seconds: Optional[float] = None, reason: str = "relocation"):
        self.time = 0.0
        self.delay = delay                  # seconds the screen stays clear (and frozen) before fading
        self.on_black = on_black
        self.applied = False
        self.done = False
        self.reason = reason
        if out_seconds is not None:
            self.OUT_SECONDS = out_seconds
        if hold_seconds is not None:
            self.HOLD_SECONDS = hold_seconds
        if in_seconds is not None:
            self.IN_SECONDS = in_seconds

    @property
    def alpha(self) -> float:
        t = self.time - self.delay
        if t <= 0.0:
            return 0.0
        if t < self.OUT_SECONDS:
            return t / self.OUT_SECONDS
        if t < self.OUT_SECONDS + self.HOLD_SECONDS:
            return 1.0
        t -= self.OUT_SECONDS + self.HOLD_SECONDS
        return max(0.0, 1.0 - t / self.IN_SECONDS)

    @property
    def black(self) -> bool:
        return self.time - self.delay >= self.OUT_SECONDS

    def update(self, delta_time: float) -> None:
        self.time += delta_time
        if not self.applied and self.black:
            self.applied = True
            self.on_black()
        if self.time - self.delay >= self.OUT_SECONDS + self.HOLD_SECONDS + self.IN_SECONDS:
            self.done = True


class LongTask:
    """Work too long for one frame - days waited out in the cells, a night's sleep, an hour's wait:
    ``steps`` is a generator doing a little at a time and yielding how far it has got (0 .. 1).
    ``advance`` does as much as ``budget`` seconds allow (``None``: all of it), so the app keeps
    drawing - a box with the ``label`` and a bar - while the hours pass."""

    def __init__(self, label: str, steps, on_done: Optional[Callable[[], None]] = None):
        self.label = label
        self._steps = steps
        self.on_done = on_done
        self.progress = 0.0
        self.done = False
        self.started = time.perf_counter()

    SHOW_AFTER = 0.25            # seconds before the app bothers with a box (an hour's wait is over sooner)

    @property
    def worth_showing(self) -> bool:
        return time.perf_counter() - self.started >= self.SHOW_AFTER

    def advance(self, budget: Optional[float] = None) -> bool:
        start = time.perf_counter()
        while not self.done:
            try:
                self.progress = max(self.progress, float(next(self._steps)))
            except StopIteration:
                self.done = True
                self.progress = 1.0
                break
            if budget is not None and time.perf_counter() - start >= budget:
                break
        return self.done

    def finish(self) -> None:
        if self.on_done is not None:
            callback, self.on_done = self.on_done, None
            callback()


class Game:
    def __init__(self, level: Optional[LevelDefinition] = None, settings: GameSettings = DEFAULT_SETTINGS,
                 save_file: Optional[str] = None, messages: Optional[MessageLog] = None, tutorial: bool = False):
        self.settings = settings
        self.messages = messages or MessageLog(settings.max_log_messages,
                                               echo=stdout_echo if settings.echo_messages_to_stdout else None)
        self.rng = random.Random(settings.random_seed)   # one RNG for every gameplay roll and phrasing (seedable)
        self.world = World(level or get_level("town"), settings, notify_callback=self.messages.add, rng=self.rng)
        if hasattr(self.messages, "news"):
            self.world.news_callback = self.messages.news     # (the town's comings and goings: kept, never popped up)
        self.player: Player = self.world.spawn_player()
        self.saves = SaveManager(settings.resolved_save_file(save_file), notify_callback=self.messages.add)
        # The lock mini-game draws its sweet spots from a stream of its own so that scripted runs can
        # predict them (the town's idle chatter would otherwise shuffle the world RNG unpredictably).
        self.lockpick_rng = random.Random(settings.random_seed)

        self.lockpicking = LockpickMinigame(settings.lockpick, rng=self.lockpick_rng, rewards=settings.skill_rewards,
                                            progression_settings=settings.progression)
        self.transfer: Optional[TransferSession] = None
        self.transfer_mode: GameMode = GameMode.EXPLORE
        self.target: Optional[InteractionTarget] = None
        self.confirmation: Optional[Confirmation] = None
        self.fade: Optional[ScreenFade] = None
        # something taking many frames (``LongTask``): with ``long_task_budget`` None (headless, the
        # tests) it is done there and then; the app sets a budget per frame and draws its progress
        self.long_task: Optional[LongTask] = None
        self.long_task_budget: Optional[float] = None
        self.pending_relocations: list = []
        self.world.relocation_handler = self._on_relocation
        self.world.caught_handler = self._on_caught

        self.running = True
        self.movement = MovementResult()
        self.frame_count = 0
        # the World Events log is hidden until asked for ([`] - shown as [~]); meanwhile what concerns
        # the player pops up for a few seconds instead (``toasts``: (text, until), newest last)
        self.show_event_log = False
        self.toasts: List[Tuple[str, float]] = []
        self.ui_time = 0.0                  # real seconds of play, for the pop-ups
        self._toast_serial = getattr(self.messages, "serial", 0)
        # the [Tab] equipment view: a cursor over the inventory, the figure turning on its stand
        self.equipment_open = False
        self.equipment_cursor = 0
        self.equipment_scroll = 0
        self.preview_yaw = 0.0
        self.preview_spin = True            # the figure turns slowly by itself until the player turns it
        self.reading = None                 # (readable, its lines) on screen while the player reads it
        self.read_already: set = set()      # what has been read once (its words went into the log)

        if self.world.level.intro_message and not tutorial:
            self.messages.add(self.world.level.intro_message)
        self.quests = QuestLog(self)
        self.tutorial: Optional[Tutorial] = None
        if tutorial:
            self.tutorial = Tutorial(self)
            self.tutorial.begin()
        elif self.world.level.town_gate:
            self.world.open_gate(self.world.level.town_gate)      # no tutorial to finish: the town gate stands open

    # ------------------------------------------------------------------ the tutorial's leash
    @property
    def in_tutorial(self) -> bool:
        return self.tutorial is not None and self.tutorial.active

    @property
    def tutorial_hint(self) -> str:
        return self.tutorial.hint if self.tutorial is not None else ""

    @property
    def tutorial_waypoint(self):
        """Where the tutorial's arrow points (a world position), or None."""
        return self.tutorial.waypoint if self.tutorial is not None and self.tutorial.active else None

    # the guide panel and its arrow: the tutorial while it runs, then the quest in hand
    @property
    def guide_title(self) -> str:
        if self.tutorial_hint:
            return "TUTORIAL"
        title = self.quests.title
        return f"QUEST: {title}" if title else ""

    @property
    def guide_hint(self) -> str:
        return self.tutorial_hint or self.quests.hint

    @property
    def waypoint(self):
        """Where the guide's arrow points: the tutorial's next step, else the active quest's."""
        if self.in_tutorial:
            return self.tutorial_waypoint
        return self.quests.waypoint

    def _refused(self, kind: str, obj=None) -> bool:
        """True (with the reason in the log) if the tutorial keeps the player from this right now."""
        if self.tutorial is None:
            return False
        reason = self.tutorial.refuses(kind, obj)
        if reason is None:
            return False
        self.messages.add(reason)
        return True

    def _allowed_in(self, kind: str, container=None):
        """An ``allowed`` hook for a transfer policy, or None when nothing is restricted."""
        if self.tutorial is None or not self.tutorial.locked:
            return None
        if kind == "item":
            return lambda item: self.tutorial.refuses("item", (container, item))
        return lambda item: self.tutorial.refuses(kind, item)

    # ------------------------------------------------------------------ queries
    @property
    def mode(self) -> GameMode:
        if self.confirmation is not None:
            return GameMode.CONFIRM
        if self.lockpicking.active:
            return GameMode.LOCKPICK
        if self.transfer is not None:
            return self.transfer_mode
        if self.equipment_open:
            return GameMode.EQUIPMENT
        return GameMode.EXPLORE

    @property
    def in_equipment(self) -> bool:
        return self.equipment_open

    @property
    def in_transition(self) -> bool:
        """A fade to black is running: the player is frozen until it has played out."""
        return self.fade is not None

    @property
    def frozen(self) -> bool:
        """Controls locked: just caught by a guard (a moment to take it in), or mid-fade."""
        return self.player.frozen or self.fade is not None or self.long_task is not None

    @property
    def fade_alpha(self) -> float:
        return self.fade.alpha if self.fade is not None else 0.0

    @property
    def in_lockpicking(self) -> bool:
        return self.lockpicking.active

    @property
    def in_transfer(self) -> bool:
        return self.transfer is not None

    @property
    def in_trade_menu(self) -> bool:
        return self.transfer is not None and self.transfer_mode is GameMode.TRADE

    @property
    def active_merchant(self):
        return self.transfer.policy.merchant if self.in_trade_menu else None

    @property
    def mouse_captured(self) -> bool:
        """Whether the window should hide+grab the mouse (free look). Released in menus or after quitting."""
        return self.running and self.transfer is None and self.confirmation is None and not self.equipment_open

    @property
    def accepts_free_look(self) -> bool:
        return (self.transfer is None and not self.lockpicking.active and self.confirmation is None and not self.frozen
                and not self.equipment_open)

    @property
    def in_explore_mode(self) -> bool:
        return self.mode is GameMode.EXPLORE

    @property
    def interaction_prompt(self) -> Optional[str]:
        return self.target.prompt if self.target else None

    @property
    def interaction_title(self) -> Optional[str]:
        """The name of what the player is looking at, shown over the prompt (a door's: "Town Gate")."""
        return self.target.title if self.target else None

    @property
    def ambient_light(self) -> float:
        return self.world.ambient_light

    # ------------------------------------------------------------------ per-frame
    TARGET_OCCLUDER_REACH = 10.0     # walls farther than this cannot stand between the player and what they reach for

    def update(self, delta_time: float, move_intent: MoveIntent = MoveIntent()) -> None:
        """Advance the simulation one frame. Call after all input actions for the frame. With the
        equipment view open the world stands still: only the figure on its stand turns."""
        self.frame_count += 1
        self._update_toasts(delta_time)
        if self.long_task is not None:
            if not self.long_task.advance(self.long_task_budget):
                self.movement = MovementResult()
                return                                   # (the hours are passing: nothing else moves)
            task, self.long_task = self.long_task, None
            task.finish()
            delta_time = 0.0                             # (the frame that finishes it carries on from there)
        if self.equipment_open:
            if self.preview_spin:
                self.preview_yaw = (self.preview_yaw + self.PREVIEW_SPIN_SPEED * delta_time) % 360.0
            self.movement = MovementResult()
            return
        self.player.update_stance(delta_time)
        self.player.noise_skill_multiplier = noise_multiplier(self.player, self.settings.stealth, self.settings.progression)
        self.lockpicking.update(delta_time)
        self.player.update_turning(delta_time)
        if self.fade is not None:
            self.fade.update(delta_time)
            if self.fade.done:
                self.fade = None
                if self.pending_relocations:
                    # a move requested while some other fade was running and never applied by it:
                    # give it a fade of its own (``World.relocating`` must not stay on for good)
                    self.fade = ScreenFade(self._apply_relocations)
        if self.accepts_free_look:
            p = self.player.position
            x0, z0, y0 = p.x, p.z, p.y
            self.movement = apply_movement(self.player, move_intent, delta_time, self.settings.player,
                                           self.world.colliders(near=p))
            if (p.x, p.z) != (x0, z0) and not self.world.can_step(x0, z0, y0, p.x, p.z):
                # too steep, deep water, the edge of the world: slide along it if half the step will do
                x1, z1 = p.x, p.z
                if self.world.can_step(x0, z0, y0, x1, z0):
                    p.x, p.z = x1, z0
                elif self.world.can_step(x0, z0, y0, x0, z1):
                    p.x, p.z = x0, z1
                else:
                    p.x, p.z = x0, z0
            p.y = self.world.floor_height_at(p.x, p.z, p.y)          # up or down the steps under our feet
        else:
            self.movement = MovementResult()

        self.world.update_time(delta_time)
        self.world.update_npcs(delta_time)
        self.world.update_perception(self.player, self.movement.is_moving, self.movement.is_sprinting, delta_time)
        if self.tutorial is not None:
            self.tutorial.update(delta_time)
        self.quests.update(delta_time)
        doors, npcs, containers, stations, beds = self.world.interaction_candidates(self.player.position)
        self.target = find_interaction_target(self.player, doors, npcs,
                                              self.settings.interaction, containers,
                                              self.world.walls_near(self.player.position, self.TARGET_OCCLUDER_REACH),
                                              stations, self.settings,
                                              beds=beds, may_sleep_in=self.world.player_may_sleep_in,
                                              hour=self.world.clock.hour, readables=self.world.readables_near(self.player.position))
        if self.reading is not None and (self.reading[0].center.distance_to(self.player.position) > self.READING_REACH
                                         or not self.in_explore_mode):
            self.reading = None                                       # walked away from it

    # ------------------------------------------------------------------ speech bubbles
    HEAR_RADIUS = 12.0          # how far away talk can be made out
    FACING_DOT = 0.3            # cos(~72 deg): the speaker must be roughly in front of the player

    def speech_bubbles(self):
        """(npc, Speech) for everyone currently talking whom the player can hear - close enough
        and roughly in front of them (you hear people you face, even through a wall). The HUD
        draws these above the speakers' heads; the log keeps every line regardless."""
        from .core.math3d import view_direction
        now = self.world.elapsed
        p = self.player.position
        vx, _, vz = view_direction(self.player.rotation_y, 0.0)
        out = []
        for npc in self.world.npcs:
            speech = npc.current_speech(now)
            if speech is None:
                continue
            dx, dz = npc.position.x - p.x, npc.position.z - p.z
            dist = (dx * dx + dz * dz) ** 0.5
            if dist > self.HEAR_RADIUS:
                continue
            if dist > 1e-6 and (vx * dx + vz * dz) / dist < self.FACING_DOT:
                continue
            out.append((npc, speech))
        return out

    # ------------------------------------------------------------------ relocation & being caught
    def _on_relocation(self, request: Relocation) -> None:
        """The world wants the player somewhere else: fade to black, move them, fade back in.
        Being marched off to jail waits a couple of seconds first so the sentence sinks in."""
        self.close_screens()
        self.pending_relocations.append(request)
        if self.fade is None:
            delay = self.settings.justice.jail_fade_delay if request.reason == "jailed" else 0.0
            self.fade = ScreenFade(self._apply_relocations, delay=delay)
        elif self.fade.applied:
            self._apply_relocations()                  # already black: apply straight away

    def _apply_relocations(self) -> None:
        for request in self.pending_relocations:
            request.apply(self.player)
        self.pending_relocations.clear()
        self.world.push_out_of_doors(self.player)

    def _on_caught(self, npc) -> None:
        """An NPC has the player bang to rights: every mini-game / pocket screen closes (the
        world has already locked the controls and started the turn towards them)."""
        self.close_screens()

    def close_screens(self) -> None:
        """Shut the lockpicking mini-game, any transfer screen, the equipment view and any question."""
        if self.lockpicking.active:
            self.lockpicking.cancel()
        if self.transfer is not None:
            self.close_transfer()
        self.equipment_open = False
        self.confirmation = None
        self.reading = None

    # ------------------------------------------------------------------ the [Tab] equipment view
    PREVIEW_SPIN_SPEED = 24.0          # degrees per second the figure turns on its own

    def tab(self) -> None:
        """[Tab]: switch columns in a transfer screen; otherwise open (or close) the equipment view."""
        if self.transfer is not None:
            self.transfer.switch_side()
        elif self.equipment_open:
            self.close_equipment()
        else:
            self.open_equipment()

    def open_equipment(self) -> bool:
        """Look at yourself: the inventory with a cursor and your figure wearing what is equipped.
        Only from the explore mode (never over a screen, a question or while frozen)."""
        if not self.in_explore_mode or self.frozen:
            return False
        self.equipment_open = True
        self.preview_spin = True
        self._clamp_equipment_cursor()
        if self.tutorial is not None:
            self.tutorial.tab_opened = True
        return True

    def close_equipment(self) -> None:
        self.equipment_open = False

    def equipment_rows(self) -> list:
        return list(self.player.inventory)

    def _clamp_equipment_cursor(self) -> None:
        self.equipment_cursor = max(0, min(self.equipment_cursor, len(self.player.inventory) - 1))

    @property
    def equipment_selection(self):
        """The inventory item under the cursor in the equipment view (None if the bag is empty)."""
        self._clamp_equipment_cursor()
        rows = self.player.inventory
        return rows[self.equipment_cursor] if 0 <= self.equipment_cursor < len(rows) else None

    def equipment_move(self, delta: int) -> int:
        count = len(self.player.inventory)
        if count:
            self.equipment_cursor = (self.equipment_cursor + delta) % count
        return self.equipment_cursor

    def equipment_visible(self, rows_shown: int):
        """(first index, items shown) for the equipment view's list, keeping the cursor in view."""
        rows = self.player.inventory
        self._clamp_equipment_cursor()
        max_scroll = max(0, len(rows) - rows_shown)
        first = min(self.equipment_scroll, max_scroll)
        if self.equipment_cursor < first:
            first = self.equipment_cursor
        elif self.equipment_cursor >= first + rows_shown:
            first = self.equipment_cursor - rows_shown + 1
        self.equipment_scroll = max(0, min(first, max_scroll))
        return self.equipment_scroll, rows[self.equipment_scroll:self.equipment_scroll + rows_shown]

    def equipment_toggle(self) -> Optional[bool]:
        """[Enter] in the equipment view: put on / take off the selected item. Returns the new
        worn state, or None if the thing cannot be worn."""
        item = self.equipment_selection
        if item is None:
            return None
        result = self.player.toggle_equipped(item)
        if result is None:
            self.messages.add(f"You cannot wear or wield {item.display_name}.")
        elif result:
            self.messages.add(f"You put on {item.name}." if item.gear not in ("dagger", "jeweled_dagger", "sword", "sling")
                              else f"You take up {item.name}.")
        else:
            self.messages.add(f"You take off {item.name}." if item.gear not in ("dagger", "jeweled_dagger", "sword", "sling")
                              else f"You put away {item.name}.")
        return result

    def rotate_preview(self, steps: int) -> float:
        """[Left]/[Right] in the equipment view turn the figure (and stop it turning by itself)."""
        self.preview_spin = False
        self.preview_yaw = (self.preview_yaw + steps * 15.0) % 360.0
        return self.preview_yaw

    # ------------------------------------------------------------------ confirmations
    def ask(self, text: str, on_yes: Callable[[], None], on_no: Optional[Callable[[], None]] = None,
            yes_label: str = "[ENTER] Yes", no_label: str = "[ESC] No",
            alt_label: Optional[str] = None, on_alt: Optional[Callable[[], None]] = None) -> Confirmation:
        self.confirmation = Confirmation(text, on_yes, on_no, yes_label, no_label, alt_label, on_alt)
        return self.confirmation

    def confirm_yes(self) -> bool:
        if self.confirmation is None:
            return False
        question, self.confirmation = self.confirmation, None
        question.on_yes()
        return True

    def confirm_no(self) -> bool:
        if self.confirmation is None:
            return False
        question, self.confirmation = self.confirmation, None
        if question.on_no:
            question.on_no()
        return True

    def confirm_alt(self) -> bool:
        if self.confirmation is None or self.confirmation.on_alt is None:
            return False
        question, self.confirmation = self.confirmation, None
        question.on_alt()
        return True

    # ------------------------------------------------------------------ global actions
    def request_quit(self) -> None:
        self.running = False

    def escape(self) -> bool:
        """ESC: back out of the current screen (a question, the mini-game, a transfer).
        Returns False when there was nothing to back out of - the app then opens its pause menu."""
        if self.confirmation is not None:
            self.confirm_no()
        elif self.lockpicking.active:
            self.lockpicking.cancel()
        elif self.transfer is not None:
            self.close_transfer()
        elif self.equipment_open:
            self.close_equipment()
        elif self.reading is not None:
            self.reading = None
        else:
            return False
        return True

    def look(self, dx_pixels: float, dy_pixels: float) -> None:
        if self.accepts_free_look:
            self.player.look(dx_pixels, dy_pixels)

    # The following world actions are only available while exploring (not inside menus).
    @property
    def save_refusal(self) -> Optional[str]:
        """Why the game cannot be saved at this moment (None when it can): inside a screen, while
        the controls are locked or the screen is fading, or while the world has the player in
        transit (on the way to the cells: ``World.save_blocked``)."""
        if not self.in_explore_mode:
            return "Close the screen first."
        if self.world.save_blocked is not None:
            return self.world.save_blocked
        if self.frozen:
            return "Wait a moment."
        return None

    @property
    def can_save(self) -> bool:
        return self.save_refusal is None

    def save_game(self, path: Optional[str] = None) -> bool:
        """Write a save (to ``path`` or the quick-save file). Refused - with a line in the log -
        while a save would catch the player mid-transition."""
        refusal = self.save_refusal
        if refusal is not None:
            self.messages.add(f"You cannot save right now. {refusal}")
            return False
        extra = {"tutorial": self.tutorial.to_dict()} if self.tutorial is not None else {}
        extra["quests"] = self.quests.to_dict()
        extra["read"] = sorted(self.read_already)
        return self.saves.save(self.world, self.player, path, extra=extra) is not None

    def load_game(self, path: Optional[str] = None) -> bool:
        """Load a save into this game. Everything that belonged to the running moment - an open
        screen, a fade, a locked camera, a relocation still in flight, the town's conversations
        and every NPC's short-term state - is dropped first (``World.reset_transient``), and the
        loaded state is checked straight away (``update(0)``: jail and curfew rules, the crosshair
        target, perception) so nothing waits for the next frame or the next hour."""
        if not self.in_explore_mode:
            self.messages.add("You cannot load right now. Close the screen first.")   # (as for saving: never silently)
            return False
        self.close_screens()
        self.fade = None
        self.long_task = None                            # (hours still passing in the old session: dropped)
        self.pending_relocations.clear()
        self.player.freeze_timer = 0.0
        self.player.turn_target = None
        if not self.saves.load(self.world, self.player, path):
            return False
        data = self.saves.last_loaded or {}
        self.tutorial = Tutorial.from_dict(self, data["tutorial"]) if data.get("tutorial") else None
        if self.tutorial is None:
            self.world.suppress_curfews = False
            self.world.unfreeze_guards()
        self.quests.load_dict(data.get("quests"))
        self.read_already = set(data.get("read", ()))
        self.reading = None
        gate = self.world.level.town_gate
        if gate and not self.in_tutorial and not any(d.door_id in {x.get("door_id") for x in data.get("doors", [])}
                                                     for d in self.world.doors if d.gate == gate):
            self.world.open_gate(gate)                   # a save from before the town had a gate: it stands open
        self.movement = MovementResult()
        self.target = None
        self.update(0.0)
        return True

    def toggle_torch(self) -> Optional[bool]:
        if not self.in_explore_mode:
            return None
        self.player.has_lit_torch = not self.player.has_lit_torch
        self.messages.add(f"Torch is now {'LIT' if self.player.has_lit_torch else 'EXTINGUISHED'}.")
        return self.player.has_lit_torch

    def toggle_crouch(self) -> Optional[bool]:
        if not self.in_explore_mode:
            return None
        self.player.is_crouching = not self.player.is_crouching
        self.messages.add(f"Stance: {'Crouching' if self.player.is_crouching else 'Standing'}.")
        return self.player.is_crouching

    def wait(self, hours: int = 1) -> bool:
        """[T]: pass an hour standing still (NPCs carry on with their day). A prisoner is
        first asked whether to wait out the whole sentence instead."""
        if self.confirmation is not None and self.confirmation.on_alt is not None:
            return self.confirm_alt()                    # [T] again inside the jail question: just the hour
        if not self.in_explore_mode or self.frozen:
            return False
        jail = self.world.jail
        if jail.sealed:
            days = jail.days_left(self.world.day_count)
            self.ask(f"Wait out your sentence ({days} day{'s' if days != 1 else ''} left)?",
                     on_yes=self.wait_out_sentence, on_no=None, yes_label="[ENTER] Serve it all",
                     no_label="[ESC] Cancel", alt_label="[T] Just one hour", on_alt=lambda: self._wait_hours(1))
            return True
        return self._wait_hours(hours)

    def _wait_hours(self, hours: int) -> bool:
        """Pass the hours standing still - with the town's eyes open: a keeper closing up, a
        resident at curfew or a guard with a grudge cuts the wait short."""
        if self.tutorial is not None:
            self.tutorial.note_wait()
        self.messages.add(f"You wait for {hours} hour{'s' if hours != 1 else ''}...")
        result = {"waited": 0.0}

        def steps():
            for waited in self.world.waiting(hours, player=self.player):
                result["waited"] = waited
                yield waited / hours

        def done() -> None:
            if self.world.wait_interruption is not None:
                self.messages.add("Your wait is interrupted.")
        self.start_long_task("Waiting", steps(), done)
        return self.long_task is not None or result["waited"] > 0.0

    def wait_out_sentence(self, max_hours: Optional[int] = None) -> int:
        """Pass time hour by hour until the prisoner is released (or breaks out) - however many days
        that is: the first hour in full, the rest quick hours (``World.pass_hour``), under a progress
        box in the app. Returns the hours waited (0 while it is still going on, in the app)."""
        if self.tutorial is not None:
            self.tutorial.note_wait()
        self.messages.add("You settle down to wait out your sentence...")
        jail, world = self.world.jail, self.world
        to_go = max(1, (jail.days_left(world.day_count) * 24 + self.settings.justice.release_hour - world.clock.hour))
        limit = max_hours if max_hours is not None else to_go + 48          # (a safety net, never the sentence's length)
        result = {"waited": 0}

        def steps():
            while world.jail.sealed and result["waited"] < limit:
                world.pass_hour(quick=result["waited"] > 0)           # (the first hour lived in full)
                result["waited"] += 1
                yield min(0.99, result["waited"] / to_go)

        self.start_long_task("Serving your sentence", steps())
        return result["waited"]

    def report_crime(self, crime_name: str, exclude=()):
        """Let the NPCs (but ``exclude``) try to witness a crime the player is committing right now.
        Returns the witnesses."""
        return self.world.report_crime_to_witnesses(crime_name, self.player.position, exclude=exclude)

    # ------------------------------------------------------------------ interaction
    def interact(self) -> None:
        """[E]: close an open menu, otherwise act on the current target.
        Ignored during the lockpicking mini-game."""
        if self.transfer is not None:
            self.close_transfer()
            return
        if self.reading is not None and not self.lockpicking.active and self.confirmation is None:
            self.reading = None                  # (put down what is being read, wherever the player is looking)
            return
        if self.lockpicking.active or self.confirmation is not None or self.equipment_open or self.frozen or self.target is None:
            return
        if self.target.is_door:
            if not self._refused("door", self.target.obj):
                self._interact_door(self.target.obj)
        elif self.target.is_container:
            if not self._refused("container", self.target.obj):
                self._interact_container(self.target.obj)
        elif self.target.is_station:
            if not self._refused("station", self.target.obj):
                use_station(self.player, self.target.obj, self.settings, self.messages.add, rng=self.rng)
        elif self.target.is_npc:
            self._interact_npc(self.target.obj, self.target.npc_action)
        elif self.target.is_bed:
            self.sleep(self.target.obj)
        elif self.target.is_readable:
            self.read(self.target.obj)

    # ------------------------------------------------------------------ reading
    READING_REACH = 4.0          # walk further than this from what you are reading and you stop

    def read(self, readable) -> None:
        """[E] on a notice, a signpost, a journal: its words on screen until [E] again or the
        player walks off (the first time, they go into the log too)."""
        if self.reading is not None and self.reading[0] is readable:
            self.reading = None                                       # (put it down)
            return
        self.reading = (readable, list(readable.lines))
        if readable.name not in self.read_already:
            self.read_already.add(readable.name)
            self.messages.add(f"{readable.label}: " + " ".join(readable.lines))

    # ------------------------------------------------------------------ sleeping
    WAKE_HOUR = 6              # "Sleep Until Morning": the sun is up
    NIGHTFALL_HOUR = 20        # "Sleep Until Night": full dark (``DayNightCycle.is_night``), when the streets empty

    @classmethod
    def sleep_target(cls, hour: int):
        """(hour to wake at, name of that time) for a sleep begun at ``hour``: by day a thief
        sleeps until night, at night (or dusk) until morning."""
        return (cls.NIGHTFALL_HOUR, "night") if cls.WAKE_HOUR <= hour < 18 else (cls.WAKE_HOUR, "morning")

    def sleep(self, bed=None) -> bool:
        """Sleep in a bed that is the player's (their own, or the room they are renting): the
        screen fades slowly to black, the hours pass (the town lives on) and it is 20:00 -
        nightfall - when it comes back if you lay down by day, 06:00 if you lay down at night.
        The same for every bed the player may use."""
        if bed is not None and not self.world.player_may_sleep_in(bed):
            self.messages.add("That is not your bed.")
            return False
        if self.fade is not None:
            return False
        self.close_screens()
        wake_hour, when = self.sleep_target(self.world.clock.hour)
        self.messages.add(f"You settle down to sleep until {when}...")
        self.fade = self._sleep_fade(wake_hour)
        return True

    def sleep_until_morning(self, bed=None) -> bool:
        """Lie down for the night, whatever the hour (the bed action by day sleeps until night)."""
        if bed is not None and not self.world.player_may_sleep_in(bed):
            self.messages.add("That is not your bed.")
            return False
        if self.fade is not None:
            return False
        self.close_screens()
        self.messages.add("You settle down to sleep until morning...")
        self.fade = self._sleep_fade(self.WAKE_HOUR)
        return True

    def _sleep_fade(self, wake_hour: int) -> ScreenFade:
        """The slow fade of lying down. At black, whatever move the world asked for while the screen
        was still darkening (checkout struck as the guest lay down) is made first - that fade is
        this one - and then the hours pass."""
        def at_black() -> None:
            self._apply_relocations()
            self._sleep_until(wake_hour)
        return ScreenFade(at_black, out_seconds=1.8, hold_seconds=0.8, in_seconds=1.4, reason="sleep")

    def _sleep_until(self, hour: int) -> None:
        def woken() -> None:
            self.messages.add(f"You wake at {self.world.clock.hour:02d}:00, rested.")
        self.start_long_task("Sleeping", self.world.sleeping_until(hour), woken, then_update=False)

    def start_long_task(self, label: str, steps, on_done: Optional[Callable[[], None]] = None, then_update: bool = True) -> None:
        """Run ``steps`` (see ``LongTask``): straight away without a per-frame budget (headless),
        otherwise over the frames to come, ``update`` doing a budget's worth each. ``then_update``
        (done straight away): a zero-length frame afterwards, so the target, the prompt and the
        rest are up to date."""
        task = LongTask(label, steps, on_done)
        if self.long_task_budget is None:
            task.advance(None)
            task.finish()
            if then_update:
                self.update(0.0)
            return
        self.long_task = task

    @property
    def busy_label(self) -> Optional[str]:
        """What the long task under way is called (for the box the app draws), if there is one."""
        return self.long_task.label if self.long_task is not None else None

    def _interact_container(self, container) -> None:
        if not getattr(container, "is_locked", False):
            self.open_container(container)
        elif container.key_id and self.player.has_key_for(container.key_id):
            container.unlock_and_open()
            self.messages.add(f"Unlocked the {container.label.lower()} with your key.")
        elif self.player.is_crouching and self.player.has_lockpick():
            self._begin_lockpicking(container, f"Started lockpicking the {container.label.lower()} "
                                               f"({self._lock_words(container)}).", "Lockpicking a lockbox")
        elif self.player.is_crouching:
            self.messages.add(f"The {container.label.lower()} is locked! You need a key or a lockpick.")
        else:
            self.messages.add(f"The {container.label.lower()} is locked. Crouch to pick it, or find the key.")

    def _begin_lockpicking(self, lockable, message: str, crime: str) -> None:
        """The lock's tier against the player's effective Lockpicking sets the window, the wobble
        and the odds the pins set (``systems/stealth.py``)."""
        from .systems.stealth import lock_challenge, lock_tier_name, lockpick_chance
        st, pr = self.settings.stealth, self.settings.progression
        self.lockpicking.begin(lockable, tolerance=lockpick_tolerance(self.player, lockable, st, pr),
                               fail_xp=self.settings.skill_rewards.lockpicking_fail,
                               wobble=lockpick_wobble(self.player, lockable, st, pr),
                               set_chance=lockpick_chance(self.player, lockable, st, pr),
                               challenge=lock_challenge(lockable, st), tier_name=lock_tier_name(lockable))
        self.report_crime(crime)
        self.messages.add(message)

    def _lock_words(self, lockable) -> str:
        """'a Sturdy lock, 64% at your skill' - what the player is up against."""
        from .systems.stealth import lock_tier_name, lockpick_chance
        odds = lockpick_chance(self.player, lockable, self.settings.stealth, self.settings.progression)
        return f"a {lock_tier_name(lockable)} lock, {odds:.0%} at your skill"

    def _interact_door(self, door) -> None:
        if not door.is_locked:
            door.toggle()
            self.world.push_out_of_doors(self.player)
        elif self.player.has_key_for(door.door_id):
            door.is_locked = False
            self.messages.add("Unlocked the door with your key.")
        elif self.player.is_crouching and self.player.has_lockpick():
            self._begin_lockpicking(door, f"Started lockpicking the door ({self._lock_words(door)}).", "Lockpicking a door")
        elif self.player.is_crouching:
            self.messages.add("Door is locked! You need a key or a lockpick.")
        else:
            self.messages.add("Door is locked. Crouch to pick it, or find the key.")

    def _interact_npc(self, npc, action: Optional[NPCAction]) -> None:
        if action is NPCAction.PICKPOCKET:
            if not self._refused("pickpocket", npc):
                self.open_pickpocket(npc)
        elif action is NPCAction.TOO_FAR_TO_PICKPOCKET:
            self.messages.add(f"Sneak closer to pick {npc.name}'s pockets.")
        else:
            if self.quests.talk(npc):
                return                                   # a job offered, or one handed in
            if action is NPCAction.TRADE and self._refused("trade", npc):
                return
            if action is NPCAction.TRADE and npc.is_asleep:
                npc.wake_for_business()                  # a fence: roused for a deal, back to bed afterwards
            if not npc.is_asleep:
                npc.turn_towards(self.player.position)
            npc.interact(notify_callback=self.messages.add, hour=self.world.simulated_hour)
            if action is NPCAction.TRADE:
                self.open_trade(npc)

    # ------------------------------------------------------------------ transfer sessions
    def open_trade(self, merchant) -> TransferSession:
        policy = TradePolicy(merchant, self.player, self.settings.economy, notify=self.messages.add,
                             rewards=self.settings.skill_rewards, progression_settings=self.settings.progression,
                             on_service=self._buy_service, on_refused_stolen=self._merchant_reports_fencing,
                             allowed=self._allowed_in("trade_item"))
        return self._open_transfer(self.world.merchant_stock(merchant), policy, GameMode.TRADE)

    def _merchant_reports_fencing(self, merchant) -> None:
        """An honest merchant offered stolen goods shouts for the guards: they come running and
        hold it against you (a reported crime - only the merchant saw it)."""
        self.world.declare_crime("Fencing stolen goods", self.player.position.copy(), witnessed=False, reporter=merchant.name)

    def _buy_service(self, item) -> bool:
        """Service items in a merchant's stock: a room at the inn for a night or a week."""
        from .world.inn import RENT_ITEMS
        nights = RENT_ITEMS.get(item.id)
        if nights is None or not self.world.inn.present:
            return False
        return self.world.inn.rent(self.world, self.player, nights, self.messages.add)

    def open_pickpocket(self, victim) -> TransferSession:
        """Getting into the pockets is itself a dip the bystanders may notice (a Pickpocket roll,
        like every lift that follows). The victim is not a bystander: whether *they* feel the hand
        is the other roll, made on every lift."""
        policy = PickpocketPolicy(victim, self.player, self.settings.skill_rewards, notify=self.messages.add,
                                  report_crime=lambda crime: self.report_crime(crime, exclude=(victim,)),
                                  rng=self.rng, stealth=self.settings.stealth,
                                  on_caught=self._pickpocket_caught, economy=self.settings.economy,
                                  progression_settings=self.settings.progression)
        policy.expose_to_bystanders()
        return self._open_transfer(victim.inventory, policy, GameMode.PICKPOCKET)

    def _pickpocket_caught(self, victim) -> None:
        """A fumbled pickpocket: the victim wakes up, throws the player out of their place and raises
        the alarm - once. A victim already shouting (the bystanders' roll went against the player
        moments ago) does not start again, and the bystanders were rolled for on this very lift, so
        the crime is declared with the victim as its witness rather than put to the town a second time."""
        if victim.is_asleep:
            victim.state = AIState.INVESTIGATING
            self.messages.add(f"({victim.name} woke up!): \"Hey! Get your hands off me!\"")
        self.world.player_caught(victim)
        area = self.world.area_of(victim.position)
        if area == self.world.area_of(self.player.position) and area != self.world.level.default_area:
            self.world.throw_out(self.player, area, by=victim,
                                 line=f"({victim.name}): \"Out! Get out of {area}!\" You are thrown out.")
        fresh = not victim.is_alarmed
        if fresh:
            victim.crimes_witnessed += 1
            victim.raise_alarm(self.messages.add, "A thief! Hands where I can see them!", "Thief! Guards, a pickpocket!")
        self.world.declare_crime("Pickpocketing", self.player.position.copy(), witnessed=True, reporter=victim.name,
                                 witnesses=[victim], new_witnesses=[victim] if fresh else [])

    def open_container(self, container) -> TransferSession:
        policy = ContainerPolicy(container, self.player, notify=self.messages.add, report_crime=self.report_crime,
                                 rewards=self.settings.skill_rewards, stealth=self.settings.stealth, rng=self.rng,
                                 progression_settings=self.settings.progression,
                                 complimentary=lambda item: self.world.is_complimentary(container, item),
                                 on_complimentary=lambda item: self.world.note_complimentary(container, item),
                                 allowed=self._allowed_in("item", container))
        return self._open_transfer(container.inventory, policy, GameMode.CONTAINER)

    def _open_transfer(self, left_items, policy, mode: GameMode) -> TransferSession:
        self.transfer = TransferSession(left_items, self.player.inventory, policy)
        self.transfer_mode = mode
        return self.transfer

    def close_transfer(self) -> None:
        merchant = self.active_merchant
        self.transfer = None
        self.transfer_mode = GameMode.EXPLORE
        if merchant is not None:
            merchant.back_to_business_as_usual(self.world, self.messages.add)   # a fence woken for the deal turns in again

    def transfer_switch_side(self) -> None:
        """(Kept for callers that mean the transfer column specifically; the key goes through ``tab``.)"""
        if self.transfer:
            self.transfer.switch_side()

    def loot_all(self) -> Optional[list]:
        """[X] in a container screen: take everything in it (a shelf, a table, a lockbox). Ignored
        in every other screen - somebody's pockets are emptied one dip at a time."""
        if self.transfer is None or self.transfer_mode is not GameMode.CONTAINER or self.confirmation is not None:
            return None
        if self._refused("loot_all", self.transfer.policy.container):
            return None
        results = self.transfer.take_all()
        if results and any(r.close_session for r in results):
            self.close_transfer()
        return results

    # Up/Down: move the menu cursor inside a transfer screen, scroll the inventory panel otherwise.
    def cursor_up(self) -> None:
        if self.transfer:
            self.transfer.move_selection(-1)
        elif self.equipment_open:
            self.equipment_move(-1)

    def cursor_down(self) -> None:
        if self.transfer:
            self.transfer.move_selection(+1)
        elif self.equipment_open:
            self.equipment_move(+1)

    # ------------------------------------------------------------------ the World Events log
    TOAST_SECONDS = 5.0          # how long a line that concerns the player pops up while the log is hidden
    MAX_TOASTS = 3

    def toggle_event_log(self) -> bool:
        """[`]: show or hide the World Events log (hidden at the start)."""
        self.show_event_log = not self.show_event_log
        self.toasts.clear()
        return self.show_event_log

    def _update_toasts(self, delta_time: float) -> None:
        self.ui_time += delta_time
        serial = getattr(self.messages, "serial", None)
        if serial is None:
            return
        if serial != self._toast_serial:
            if not self.show_event_log:
                for text, kind in self.messages.since(self._toast_serial):
                    if kind != MessageLog.NEWS:
                        self.toasts.append((text, self.ui_time + self.TOAST_SECONDS))
            self._toast_serial = serial
        self.toasts = [t for t in self.toasts if t[1] > self.ui_time][-self.MAX_TOASTS:]

    def transfer_confirm(self) -> Optional[TransferResult]:
        if self.confirmation is not None:
            self.confirm_yes()
            return None
        if self.equipment_open:
            self.equipment_toggle()
            return None
        if not self.transfer:
            return None
        worn = self.transfer.giving_away_worn()
        if worn is not None:
            # selling / planting / putting away something you have on: make sure first
            verb = self.transfer.policy.give_verb()
            self.ask(f"{verb} the {worn.name} you are wearing?", on_yes=self._confirm_transfer_now,
                     yes_label=f"[ENTER] {verb} it", no_label="[ESC] Keep it on")
            return None
        return self._confirm_transfer_now()

    def _confirm_transfer_now(self) -> Optional[TransferResult]:
        if not self.transfer:
            return None
        result = self.transfer.confirm()
        if result.close_session:
            self.close_transfer()
        return result

    # ------------------------------------------------------------------ lockpicking
    def lockpick_rotate(self, steps: int) -> None:
        if self.lockpicking.active:
            self.lockpicking.rotate(steps)
        elif self.equipment_open:
            self.rotate_preview(steps)

    def lockpick_attempt(self) -> Optional[LockpickOutcome]:
        if not self.lockpicking.active:
            return None
        return self.lockpicking.attempt(self.player, notify=self.messages.add)

    # ------------------------------------------------------------------ debugging aid
    def snapshot(self) -> dict:
        """Plain-data view of the full game state (used by tests and tooling)."""
        p = self.player
        return {
            "frame": self.frame_count,
            "running": self.running,
            "mode": self.mode.value,
            "hour": self.world.simulated_hour,
            "minute_accumulator": round(self.world.clock.minute_accumulator, 9),
            "player": {
                "pos": (round(p.position.x, 9), round(p.position.y, 9), round(p.position.z, 9)),
                "yaw": round(p.rotation_y, 9), "pitch": round(p.rotation_x, 9),
                "eye": round(p.current_eye_height, 6),
                "gold": round(p.gold, 9), "torch": p.has_lit_torch, "crouch": p.is_crouching,
                "inventory": [(i.id, i.quantity, i.is_stolen) for i in p.inventory],
                "skills": {k.value: dict(v) for k, v in p.skills.skills.items()},
                "crimes_witnessed": p.crimes_witnessed,
            },
            "npcs": [{"name": n.name, "state": n.state.name, "suspicion": round(n.suspicion_meter, 9),
                      "torch": n.has_torch, "inventory": [(i.id, i.quantity) for i in n.inventory], "gold": round(n.gold, 6),
                      "pos": (round(n.position.x, 6), round(n.position.y, 6), round(n.position.z, 6)),
                      "waypoints": len(n.path), "crimes": n.crimes_witnessed} for n in self.world.npcs],
            "doors": [{"id": d.door_id, "locked": d.is_locked, "open": d.is_open, "sealed": d.sealed} for d in self.world.doors],
            "containers": [{"name": c.name, "locked": c.is_locked, "inventory": [(i.id, i.quantity) for i in c.inventory]}
                           for c in self.world.containers],
            "jail": self.world.jail.to_dict(),
            "transfer": {"mode": self.transfer_mode.value, "side": self.transfer.side.value,
                         "selection": dict((s.value, v) for s, v in self.transfer.selection.items()),
                         "title": self.transfer.policy.title} if self.transfer else None,
            "lockpick": {"angle": self.lockpicking.pick_angle, "sweet_spot": self.lockpicking.sweet_spot,
                         "target": self.lockpicking.target_id},
            "confirmation": self.confirmation.text if self.confirmation else None,
            "equipment": {"open": self.equipment_open, "cursor": self.equipment_cursor,
                          "worn": {slot.value: item.id for slot, item in p.equipment.items()}},
            "tutorial": self.tutorial.stage.value if self.tutorial is not None else None,
            "fade": round(self.fade_alpha, 3),
            "frozen": self.frozen,
            "prompt": self.interaction_prompt,
            "messages": self.messages.messages,
            "moving": self.movement.is_moving, "sprinting": self.movement.is_sprinting,
            "mouse_captured": self.mouse_captured,
        }
