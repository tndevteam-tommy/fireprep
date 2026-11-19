"""Controller devices: hot-plug, event decoding and rumble (the pygame side).

Pads SDL knows (Xbox, PlayStation, Switch Pro and the ~2000 in SDL's database)
are opened through SDL's GameController API, which reports every pad in one
position-based layout.  Anything else (cheap USB pads, arcade sticks) is opened
as a raw joystick with a best-guess layout (`core.gamepad.raw_layout`).

All of them are decoded into a PadState per device and `PadEvent`s.  The most
recently used device is the *active* one - the one the game listens to - so a
second pad or a stray joystick never fights the player's controller.
"""
import os

from core.gamepad import (SDL_BUTTONS, SDL_AXES, TRIGGERS, PadState, detect_family, raw_layout,
                          FAMILY_NAMES)

# Positional face buttons on every pad (Nintendo's B is "a", the bottom button), and rumble
# over Bluetooth for PlayStation pads.  Must be set before SDL's controller subsystem starts.
os.environ.setdefault("SDL_GAMECONTROLLER_USE_BUTTON_LABELS", "0")
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS4_RUMBLE", "1")
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS5_RUMBLE", "1")

try:
    import pygame
except ImportError:          # pragma: no cover - the game needs pygame, the pure tests do not
    pygame = None

ACTIVATE_STICK = 0.5          # a stick pushed this far makes its pad the active one
_IGNORED_PAD_EVENTS = tuple(getattr(pygame, name) for name in (
    "CONTROLLERDEVICEREMAPPED", "CONTROLLERTOUCHPADDOWN", "CONTROLLERTOUCHPADMOTION", "CONTROLLERTOUCHPADUP",
    "CONTROLLERSENSORUPDATE") if pygame is not None and hasattr(pygame, name))


class PadEvent:
    BUTTON = "button"          # control pressed / released (triggers included)
    ADDED = "added"
    REMOVED = "removed"
    ACTIVATED = "activated"    # another connected pad became the active one

    def __init__(self, kind, device, control=None, down=False, was_active=False):
        self.kind = kind
        self.device = device
        self.control = control
        self.down = down
        self.was_active = was_active       # REMOVED: the device was the one the game listened to

    def __repr__(self):
        return f"PadEvent({self.kind}, {self.device.name!r}, {self.control}, {self.down})"


class PadDevice:
    GAMEPAD = "gamepad"        # SDL GameController (known mapping)
    JOYSTICK = "joystick"      # raw joystick, guessed layout

    def __init__(self, instance_id, name, guid, kind, handle, joystick=None, layout=None):
        self.instance_id = instance_id
        self.name = name or "Controller"
        self.guid = guid or ""
        self.kind = kind
        self.handle = handle
        self.joystick = joystick
        self.layout = layout
        self.family = detect_family(self.name, self.guid)
        self.state = PadState()
        self.last_used = 0.0

    @property
    def family_name(self):
        return FAMILY_NAMES[self.family]

    def describe(self):
        extra = "" if self.kind == self.GAMEPAD else ", unrecognised layout"
        return f"{self.name} ({self.family_name}{extra})"

    def rumble(self, low, high, duration_ms):
        for target in (self.handle, self.joystick):
            fn = getattr(target, "rumble", None)
            if fn is None:
                continue
            try:
                return bool(fn(float(low), float(high), int(duration_ms)))
            except Exception:          # noqa: BLE001 - unsupported on this pad / platform
                continue
        return False

    def close(self):
        for target in (self.handle, self.joystick):
            fn = getattr(target, "quit", None)
            if fn is not None:
                try:
                    fn()
                except Exception:      # noqa: BLE001
                    pass


def _load_controller_module():
    try:
        import pygame._sdl2.controller as ctl
        ctl.init()
        return ctl
    except Exception as exc:           # noqa: BLE001 - old / stripped pygame builds
        print(f"[controller] SDL GameController API unavailable ({exc}); using raw joysticks")
        return None


class GamepadManager:
    def __init__(self, clock=None, use_gamecontroller=True):
        if pygame is None:
            raise RuntimeError("pygame is required for controller support")
        import time
        self.clock = clock or time.monotonic
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        self.ctl = _load_controller_module() if use_gamecontroller else None
        self.devices = {}              # instance id -> PadDevice
        self.active = None
        self.scan()

    # ---------------------------------------------------------------- devices
    def scan(self):
        """Open every connected device (the DEVICEADDED events that follow are de-duplicated)."""
        opened = []
        for index in range(pygame.joystick.get_count()):
            dev = self._open(index)
            if dev is not None:
                opened.append(dev)
        return opened

    def _is_gamepad(self, index):
        if self.ctl is None:
            return False
        try:
            return bool(self.ctl.is_controller(index))
        except Exception:              # noqa: BLE001
            return False

    def _open(self, index):
        try:
            if self._is_gamepad(index):
                handle = self.ctl.Controller(index)
                try:
                    js = handle.as_joystick()
                except Exception:      # noqa: BLE001
                    js = pygame.joystick.Joystick(index)
                iid = js.get_instance_id()
                if iid in self.devices:
                    return None
                name = getattr(handle, "name", None) or js.get_name()
                dev = PadDevice(iid, name, js.get_guid(), PadDevice.GAMEPAD, handle, joystick=js)
            else:
                js = pygame.joystick.Joystick(index)      # pygame 2: opened on construction
                iid = js.get_instance_id()
                if iid in self.devices:
                    return None
                layout = raw_layout(js.get_numbuttons(), js.get_numaxes(), js.get_numhats())
                dev = PadDevice(iid, js.get_name(), js.get_guid(), PadDevice.JOYSTICK, js, joystick=js,
                                layout=layout)
        except Exception as exc:       # noqa: BLE001 - device vanished between enumeration and open
            print(f"[controller] could not open device {index}: {exc}")
            return None
        self.devices[dev.instance_id] = dev
        dev.last_used = self.clock()
        if self.active is None:
            self.active = dev
        print(f"[controller] connected: {dev.describe()}")
        return dev

    def _remove(self, instance_id):
        dev = self.devices.pop(instance_id, None)
        if dev is None:
            return []
        events = [PadEvent(PadEvent.BUTTON, dev, c, False) for c, _ in dev.state.clear()]
        dev.close()
        print(f"[controller] disconnected: {dev.describe()}")
        was_active = self.active is dev
        events.append(PadEvent(PadEvent.REMOVED, dev, was_active=was_active))
        if was_active:
            self.active = max(self.devices.values(), key=lambda d: d.last_used, default=None)
            if self.active is not None:
                events.append(PadEvent(PadEvent.ACTIVATED, self.active))
        return events

    def _touch(self, dev, events):
        dev.last_used = self.clock()
        if self.active is not dev:
            self.active = dev
            events.insert(0, PadEvent(PadEvent.ACTIVATED, dev))

    @property
    def active_state(self):
        return self.active.state if self.active is not None else None

    def rumble(self, low, high, duration_ms):
        if self.active is None:
            return False
        return self.active.rumble(low, high, duration_ms)

    # ----------------------------------------------------------------- events
    def process(self, event):
        """Decode a pygame event. Returns a list of PadEvents, or None if it is not a pad event."""
        t = event.type
        if t in (pygame.CONTROLLERDEVICEADDED, pygame.JOYDEVICEADDED):
            dev = self._open(event.device_index)
            return [PadEvent(PadEvent.ADDED, dev)] if dev is not None else []
        if t in (pygame.CONTROLLERDEVICEREMOVED, pygame.JOYDEVICEREMOVED):
            return self._remove(event.instance_id)
        if t in _IGNORED_PAD_EVENTS:
            return []            # remaps, DualShock / DualSense touchpad surface and motion sensors

        if t in (pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERBUTTONUP, pygame.CONTROLLERAXISMOTION):
            dev = self.devices.get(event.instance_id)
            if dev is None or dev.kind != PadDevice.GAMEPAD:
                return []
            if t == pygame.CONTROLLERAXISMOTION:
                if not 0 <= event.axis < len(SDL_AXES):
                    return []
                axis = SDL_AXES[event.axis]
                value = max(-1.0, min(1.0, event.value / 32767.0))
                return self._axis(dev, axis, value)
            if not 0 <= event.button < len(SDL_BUTTONS):
                return []
            return self._button(dev, SDL_BUTTONS[event.button], t == pygame.CONTROLLERBUTTONDOWN)

        if t in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYAXISMOTION, pygame.JOYHATMOTION):
            dev = self.devices.get(event.instance_id)
            if dev is None or dev.kind != PadDevice.JOYSTICK:
                return []          # pads opened as GameControllers report through CONTROLLER* events
            layout = dev.layout
            if t == pygame.JOYAXISMOTION:
                axis = layout["axes"].get(event.axis)
                if axis is None:
                    return []
                value = float(event.value)
                if axis in TRIGGERS:
                    value = (value + 1.0) * 0.5       # raw triggers rest at -1
                return self._axis(dev, axis, value)
            if t == pygame.JOYHATMOTION:
                if event.hat != layout["hat"]:
                    return []
                hx, hy = event.value
                out = []
                for control, on in (("dpleft", hx < 0), ("dpright", hx > 0), ("dpup", hy > 0), ("dpdown", hy < 0)):
                    out.extend(self._button(dev, control, on))
                return out
            control = layout["buttons"].get(event.button)
            if control is None:
                return []
            return self._button(dev, control, t == pygame.JOYBUTTONDOWN)
        return None

    def _button(self, dev, control, down):
        events = [PadEvent(PadEvent.BUTTON, dev, c, d) for c, d in dev.state.set_button(control, down)]
        if down and events:
            self._touch(dev, events)
        return events

    def _axis(self, dev, axis, value):
        events = [PadEvent(PadEvent.BUTTON, dev, c, d) for c, d in dev.state.set_axis(axis, value)]
        if (axis in TRIGGERS and any(e.down for e in events)) or (axis not in TRIGGERS and abs(value) >= ACTIVATE_STICK):
            self._touch(dev, events)
        return events

    def close(self):
        for dev in list(self.devices.values()):
            dev.close()
        self.devices.clear()
        self.active = None
