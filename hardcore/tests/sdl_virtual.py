"""Virtual controllers for the tests, through SDL's own virtual-joystick API (SDL >= 2.24).

SDL treats a virtual joystick exactly like a plugged-in device: it gets a GUID
with the vendor/product ids we give it, a gamepad mapping if it is of the
gamepad type, and its buttons / axes arrive as the same CONTROLLER* / JOY*
pygame events a real pad produces.  That lets the tests run the real device
path (hot-plug, decoding, family detection, raw-joystick fallback) headlessly.

`available()` is False when the SDL library pygame uses cannot be reached
(e.g. a system SDL that is too old) - the tests then skip.
"""
import ctypes
import ctypes.util
import glob
import os
import sys

SDL_JOYSTICK_TYPE_UNKNOWN = 0
SDL_JOYSTICK_TYPE_GAMECONTROLLER = 1

# (name, vendor, product) of pads the game should recognise
XBOX = ("Xbox Series X Controller", 0x045E, 0x0B12)
DUALSENSE = ("DualSense Wireless Controller", 0x054C, 0x0CE6)
DUALSHOCK4 = ("PS4 Controller", 0x054C, 0x09CC)
SWITCH_PRO = ("Nintendo Switch Pro Controller", 0x057E, 0x2009)
GENERIC = ("USB Gamepad", 0x0079, 0x0011)


class _Desc(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("type", ctypes.c_uint16), ("naxes", ctypes.c_uint16),
                ("nbuttons", ctypes.c_uint16), ("nhats", ctypes.c_uint16), ("vendor_id", ctypes.c_uint16),
                ("product_id", ctypes.c_uint16), ("padding", ctypes.c_uint16), ("button_mask", ctypes.c_uint32),
                ("axis_mask", ctypes.c_uint32), ("name", ctypes.c_char_p), ("userdata", ctypes.c_void_p),
                ("Update", ctypes.c_void_p), ("SetPlayerIndex", ctypes.c_void_p), ("Rumble", ctypes.c_void_p),
                ("RumbleTriggers", ctypes.c_void_p), ("SetLED", ctypes.c_void_p), ("SendEffect", ctypes.c_void_p)]


_sdl = None


def _is_core_sdl(path):
    name = os.path.basename(path)
    return ("SDL2" in name and not any(x in name for x in ("_image", "_mixer", "_ttf", "_gfx", "_net")))


def _candidates():
    """The SDL library pygame itself loaded (it must be the *same* copy, or the virtual pads
    would live in another SDL instance), then the usual wheel locations."""
    import pygame
    import pygame.joystick  # noqa: F401  (makes sure SDL is mapped into the process)
    try:
        with open("/proc/self/maps") as maps:          # Linux: exactly what is loaded
            for line in maps:
                path = line.split()[-1] if "/" in line else ""
                if _is_core_sdl(path) and os.path.exists(path):
                    yield path
    except OSError:
        pass
    base = os.path.dirname(pygame.__file__)
    pats = [os.path.join(base, "..", "pygame.libs", "libSDL2*"), os.path.join(base, "..", "pygame_ce.libs", "libSDL2*"),
            os.path.join(base, ".dylibs", "libSDL2*"), os.path.join(base, "SDL2*.dll"), os.path.join(base, "libSDL2*")]
    for pat in pats:
        yield from sorted(p for p in glob.glob(pat) if _is_core_sdl(p))
    lib = ctypes.util.find_library("SDL2")
    if lib:
        yield lib


def _lib():
    global _sdl
    if _sdl is not None:
        return _sdl or None
    _sdl = False
    try:
        import pygame
        if pygame.get_sdl_version() < (2, 24, 0):
            return None
    except Exception:       # noqa: BLE001
        return None
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
            lib.SDL_JoystickAttachVirtualEx.argtypes = [ctypes.POINTER(_Desc)]
            lib.SDL_JoystickAttachVirtualEx.restype = ctypes.c_int
            lib.SDL_JoystickDetachVirtual.argtypes = [ctypes.c_int]
            lib.SDL_JoystickOpen.argtypes = [ctypes.c_int]
            lib.SDL_JoystickOpen.restype = ctypes.c_void_p
            lib.SDL_JoystickClose.argtypes = [ctypes.c_void_p]
            lib.SDL_JoystickInstanceID.argtypes = [ctypes.c_void_p]
            lib.SDL_JoystickInstanceID.restype = ctypes.c_int32
            lib.SDL_JoystickSetVirtualButton.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
            lib.SDL_JoystickSetVirtualAxis.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16]
            lib.SDL_JoystickSetVirtualHat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
            lib.SDL_NumJoysticks.restype = ctypes.c_int
            lib.SDL_JoystickGetDeviceInstanceID.argtypes = [ctypes.c_int]
            lib.SDL_JoystickGetDeviceInstanceID.restype = ctypes.c_int32
            lib.SDL_JoystickIsVirtual.argtypes = [ctypes.c_int]
            lib.SDL_JoystickIsVirtual.restype = ctypes.c_int
            _sdl = lib
            return lib
        except (OSError, AttributeError):
            continue
    return None


def available():
    return _lib() is not None


# SDL hat bits
HAT_UP, HAT_RIGHT, HAT_DOWN, HAT_LEFT = 1, 2, 4, 8


_RUMBLE_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16)


class VirtualPad:
    """A virtual controller. kind='gamepad' gets SDL's standard mapping (buttons in SDL order:
    a b x y back guide start ls rs lb rb up down left right; axes lx ly rx ry lt rt);
    kind='joystick' is an unrecognised device (raw joystick path)."""

    def __init__(self, spec=XBOX, kind="gamepad", naxes=6, nbuttons=15, nhats=0, record_rumble=False):
        lib = _lib()
        if lib is None:
            raise RuntimeError("SDL virtual joysticks unavailable")
        self.lib = lib
        name, vendor, product = spec
        self.name = name
        self._name_bytes = name.encode()
        desc = _Desc(version=1,
                     type=SDL_JOYSTICK_TYPE_GAMECONTROLLER if kind == "gamepad" else SDL_JOYSTICK_TYPE_UNKNOWN,
                     naxes=naxes, nbuttons=nbuttons, nhats=nhats, vendor_id=vendor, product_id=product,
                     name=self._name_bytes)
        self.rumbles = []          # (low, high) in [0, 1] as the pad received them
        if record_rumble:
            def on_rumble(_userdata, low, high):
                self.rumbles.append((low / 65535.0, high / 65535.0))
                return 0
            self._rumble_cb = _RUMBLE_FN(on_rumble)      # keep a reference: SDL holds the pointer
            desc.Rumble = ctypes.cast(self._rumble_cb, ctypes.c_void_p)
        index = lib.SDL_JoystickAttachVirtualEx(ctypes.byref(desc))
        if index < 0:
            raise RuntimeError("SDL_JoystickAttachVirtualEx failed")
        self.device_index = index
        self.handle = lib.SDL_JoystickOpen(index)
        self.instance_id = lib.SDL_JoystickInstanceID(self.handle)
        self.kind = kind

    def button(self, index, down):
        self.lib.SDL_JoystickSetVirtualButton(self.handle, index, 1 if down else 0)

    def axis(self, index, value):
        """value in [-1, 1] (triggers: -1 = released, 1 = fully pulled, as SDL's raw range)."""
        v = int(round(max(-1.0, min(1.0, value)) * 32767))
        self.lib.SDL_JoystickSetVirtualAxis(self.handle, index, max(-32768, min(32767, v)))

    def trigger(self, index, amount):
        """amount in [0, 1] -> raw axis (-1 .. 1)."""
        self.axis(index, amount * 2.0 - 1.0)

    def hat(self, index, bits):
        self.lib.SDL_JoystickSetVirtualHat(self.handle, index, bits)

    def detach(self):
        if self.handle:
            lib = self.lib
            # the device index may have shifted as other pads came and went
            for i in range(lib.SDL_NumJoysticks()):
                if lib.SDL_JoystickGetDeviceInstanceID(i) == self.instance_id:
                    lib.SDL_JoystickClose(self.handle)
                    self.handle = None
                    lib.SDL_JoystickDetachVirtual(i)
                    return
            lib.SDL_JoystickClose(self.handle)
            self.handle = None


# SDL standard-mapping indices for VirtualPad(kind="gamepad")
BTN = {"a": 0, "b": 1, "x": 2, "y": 3, "back": 4, "guide": 5, "start": 6, "leftstick": 7, "rightstick": 8,
       "leftshoulder": 9, "rightshoulder": 10, "dpup": 11, "dpdown": 12, "dpleft": 13, "dpright": 14}
AX = {"leftx": 0, "lefty": 1, "rightx": 2, "righty": 3, "lefttrigger": 4, "righttrigger": 5}


def pump(pygame_module, manager=None):
    """Let SDL deliver the virtual device changes; feed them to a GamepadManager if given.
    Returns the list of PadEvents (or raw pygame events if no manager)."""
    out = []
    for event in pygame_module.event.get():
        if manager is None:
            out.append(event)
            continue
        res = manager.process(event)
        if res:
            out.extend(res)
    return out


if __name__ == "__main__":    # quick manual check
    print("SDL virtual joysticks available:", available(), file=sys.stderr)
