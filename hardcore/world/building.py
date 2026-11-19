"""Building generator: composite climbable structures built from Obstacle boxes.

A building is a solid core with *facade layers* on selected faces.  A facade is a
0.5 m thick skin made of horizontal spandrel bands and vertical piers; the gaps
between piers are recessed windows (and one door on the ground floor).  Every
band top is a window sill, every pier top is a window head - real ledges a
climber can grip and stand on - while the recess behind them leaves no room to
mantle, so they behave as holds rather than surfaces.  Plain faces get crevasse
courses at the *same* heights (the sill and window-head lines of every storey),
running from building corner to building corner, so every row of holds wraps the
whole building: a traverse turns every corner, plain wall or facade, and the
last course is 0.8 m under the roof edge.  A cornice crowns the roof (mantle-able all round),
optional balconies protrude from plain faces and a rooftop structure adds a
second level.
"""
import config
from world.obstacle import Obstacle, MAT_WALL, MAT_WINDOW_WALL
from parkour.ledge_rail import RailType

WALL_PLASTER, WALL_BRICK, WALL_PANELS = 0, 1, 2          # BuildingSpec.style
DECOR_NONE, DECOR_GRAFFITI, DECOR_MURAL, DECOR_GRIME = 0, 1, 2, 3   # BuildingSpec.decor

SOUTH, NORTH, EAST, WEST = "south", "north", "east", "west"
FACE_NORMALS = {SOUTH: (0.0, 0.0, 1.0), NORTH: (0.0, 0.0, -1.0), EAST: (1.0, 0.0, 0.0), WEST: (-1.0, 0.0, 0.0)}


class BuildingSpec:
    def __init__(self, x, z, width, depth, floors, facade_faces=(SOUTH,), door_face=SOUTH,
                 balconies=(), rooftop=True, floor_height=3.0, name="building",
                 color=(0.42, 0.40, 0.38), facade_color=(0.50, 0.47, 0.43), trim_color=(0.58, 0.55, 0.50),
                 awning=False, fire_escape=None, scaffold=None, rooftop_size=None, style=WALL_PLASTER, decor=DECOR_NONE):
        self.x, self.z = float(x), float(z)
        self.width, self.depth = float(width), float(depth)
        self.floors = int(floors)
        self.floor_height = float(floor_height)
        self.height = self.floors * self.floor_height
        self.facade_faces = tuple(facade_faces)
        self.door_face = door_face
        self.balconies = tuple(balconies)     # (face, floor_index, offset_along_face)
        self.rooftop = rooftop
        self.rooftop_size = rooftop_size      # (w, d, h) or None for the default
        self.awning = awning                  # slab above the door
        self.fire_escape = fire_escape        # (face, offset): a landing on every floor of a plain face
        self.scaffold = scaffold              # (face, offset, length, levels): platforms every 2 m
        self.name = name
        self.color = color
        self.facade_color = facade_color
        self.trim_color = trim_color
        self.style = int(style)             # plaster / brick / panels
        self.decor = int(decor)             # none / graffiti / mural / heavy grime

    @property
    def wall_variant(self):
        """Material variant word for MAT_WALL: style + 3 * decor (0..9)."""
        return self.style + 3 * self.decor


class BuildingGenerator:
    FACADE_DEPTH = 0.5
    WINDOW_WIDTH = 1.2
    PIER_WIDTH = 1.0
    SILL_HEIGHT = 1.0        # above the floor slab
    LINTEL_HEIGHT = 2.2      # above the floor slab (door height; a window head is reachable flat-footed)
    CORNICE = 0.3
    BALCONY_DEPTH = 1.5
    BALCONY_LENGTH = 2.6
    BALCONY_THICK = 0.2
    BALCONY_RISE = 0.4       # balcony floor sits this far above the storey line

    # ----------------------------------------------------------------- helpers
    @classmethod
    def _window_slots(cls, length):
        """Start offsets of the windows along a face of the given length."""
        pitch = cls.WINDOW_WIDTH + cls.PIER_WIDTH
        n = int((length - cls.PIER_WIDTH) // pitch)
        if n <= 0:
            return []
        used = n * pitch + cls.PIER_WIDTH
        margin = (length - used) * 0.5
        return [margin + cls.PIER_WIDTH + i * pitch for i in range(n)]

    @staticmethod
    def _face_box(spec, face, along0, along1, y0, y1, thickness, color, name):
        """Box on `face` spanning [along0, along1] along the face, `thickness` into the building."""
        x, z, w, d = spec.x, spec.z, spec.width, spec.depth
        kw = dict(add_crevasses=False, name=name, climb_target=thickness < 0,
                  material=MAT_WALL, variant=spec.wall_variant)
        if thickness < 0:
            # negative thickness: the box protrudes outward from the face (awnings)
            t = -thickness
            if face == SOUTH:
                return Obstacle(x + along0, y0, z + d, along1 - along0, y1 - y0, t, color, **kw)
            if face == NORTH:
                return Obstacle(x + along0, y0, z - t, along1 - along0, y1 - y0, t, color, **kw)
            if face == EAST:
                return Obstacle(x + w, y0, z + along0, t, y1 - y0, along1 - along0, color, **kw)
            return Obstacle(x - t, y0, z + along0, t, y1 - y0, along1 - along0, color, **kw)
        if face == SOUTH:
            return Obstacle(x + along0, y0, z + d - thickness, along1 - along0, y1 - y0, thickness, color, **kw)
        if face == NORTH:
            return Obstacle(x + along0, y0, z, along1 - along0, y1 - y0, thickness, color, **kw)
        if face == EAST:
            return Obstacle(x + w - thickness, y0, z + along0, thickness, y1 - y0, along1 - along0, color, **kw)
        return Obstacle(x, y0, z + along0, thickness, y1 - y0, along1 - along0, color, **kw)

    @staticmethod
    def _face_length(spec, face):
        return spec.width if face in (SOUTH, NORTH) else spec.depth

    # ------------------------------------------------------------------- build
    @classmethod
    def build(cls, spec):
        out = []
        fd = cls.FACADE_DEPTH
        h = spec.height

        # Core block: inset by the facade depth on facade faces, plain crevasses elsewhere
        x0, z0, x1, z1 = spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth
        if SOUTH in spec.facade_faces:
            z1 -= fd
        if NORTH in spec.facade_faces:
            z0 += fd
        if EAST in spec.facade_faces:
            x1 -= fd
        if WEST in spec.facade_faces:
            x0 += fd
        plain = [FACE_NORMALS[f] for f in (SOUTH, NORTH, EAST, WEST) if f not in spec.facade_faces]
        # plain faces are the building's walls; the faces behind a facade show glass through the
        # window recesses
        glass = {FACE_NORMALS[f]: MAT_WINDOW_WALL for f in spec.facade_faces}
        # (its own crevasses are the floor-aligned courses added below, not the generic ladder)
        core = Obstacle(x0, 0.0, z0, x1 - x0, h - cls.CORNICE, z1 - z0, spec.color,
                        add_crevasses=False, crevasse_faces=plain, name=f"{spec.name}-core", climb_target=False,
                        material=MAT_WALL, variant=spec.wall_variant, face_materials=glass)
        out.append(core)
        cls._plain_wall_courses(spec, core)

        # Facades
        for face in spec.facade_faces:
            length = cls._face_length(spec, face)
            slots = cls._window_slots(length)
            door_slot = slots[len(slots) // 2] if (face == spec.door_face and slots) else None
            for n in range(spec.floors):
                base = n * spec.floor_height
                sill = base + cls.SILL_HEIGHT
                lintel = base + cls.LINTEL_HEIGHT
                # Band below the sills (ground band or spandrel from the previous lintel)
                band_y0 = 0.0 if n == 0 else base - spec.floor_height + cls.LINTEL_HEIGHT
                if n == 0 and door_slot is not None:
                    out.append(cls._face_box(spec, face, 0.0, door_slot, band_y0, sill, fd, spec.facade_color, f"{spec.name}-band{n}a"))
                    out.append(cls._face_box(spec, face, door_slot + cls.WINDOW_WIDTH, length, band_y0, sill, fd, spec.facade_color, f"{spec.name}-band{n}b"))
                else:
                    out.append(cls._face_box(spec, face, 0.0, length, band_y0, sill, fd, spec.facade_color, f"{spec.name}-band{n}"))
                # Piers between the windows
                edges = [0.0] + [v for w0 in slots for v in (w0, w0 + cls.WINDOW_WIDTH)] + [length]
                for i in range(0, len(edges), 2):
                    a, b = edges[i], edges[i + 1]
                    if b - a > 1e-3:
                        out.append(cls._face_box(spec, face, a, b, sill, lintel, fd, spec.facade_color, f"{spec.name}-pier{n}"))
            # Top band: last lintel to the roof line
            out.append(cls._face_box(spec, face, 0.0, length, (spec.floors - 1) * spec.floor_height + cls.LINTEL_HEIGHT,
                                     h - cls.CORNICE, fd, spec.facade_color, f"{spec.name}-topband"))

        # Cornice crowning the roof (its top edges are the roof edges, mantle-able all round)
        c = cls.CORNICE
        out.append(Obstacle(spec.x - c, h - c, spec.z - c, spec.width + 2 * c, c, spec.depth + 2 * c,
                            spec.trim_color, add_crevasses=False, name=f"{spec.name}-cornice"))

        # Balconies on plain faces
        for face, floor_idx, offset in spec.balconies:
            y_top = floor_idx * spec.floor_height + cls.BALCONY_RISE
            out.append(cls._balcony(spec, face, offset, y_top))

        # Rooftop structure
        if spec.rooftop:
            rw, rd, rh = spec.rooftop_size or (min(3.0, spec.width * 0.4), min(2.6, spec.depth * 0.4), 2.2)
            out.append(Obstacle(spec.x + 0.6, h, spec.z + 0.6, rw, rh, rd, spec.facade_color, add_crevasses=False, name=f"{spec.name}-rooftop",
                                material=MAT_WALL, variant=spec.style))

        # Awning over the door (a small platform to hop onto)
        if spec.awning and spec.door_face in spec.facade_faces:
            length = cls._face_length(spec, spec.door_face)
            slots = cls._window_slots(length)
            if slots:
                door = slots[len(slots) // 2]
                out.append(cls._face_box(spec, spec.door_face, door - 0.4, door + cls.WINDOW_WIDTH + 0.4,
                                         cls.LINTEL_HEIGHT + 0.35, cls.LINTEL_HEIGHT + 0.47, -1.4,
                                         spec.trim_color, f"{spec.name}-awning"))

        # Fire escape: a landing on every floor above the ground floor
        if spec.fire_escape is not None:
            face, offset = spec.fire_escape
            for n in range(1, spec.floors):
                out.append(cls._balcony(spec, face, offset, n * spec.floor_height + cls.BALCONY_RISE, length=1.8, depth=1.0,
                                        name=f"{spec.name}-fire-escape"))

        # Scaffolding: platforms every 2 m held by posts, a climbing frame against the wall
        if spec.scaffold is not None:
            face, offset, length, levels = spec.scaffold
            for lv in range(1, levels + 1):
                out.append(cls._balcony(spec, face, offset, lv * 2.0, length=length, depth=1.2, thick=0.1,
                                        name=f"{spec.name}-scaffold"))
            if face not in spec.facade_faces:
                cls._hold_above_platform(spec, core, face, levels * 2.0)
            for along in (offset + 0.1, offset + length - 0.3):
                # posts stand 1.2 m off the wall
                if face in (SOUTH, NORTH):
                    px = spec.x + along
                    pz = spec.z + spec.depth + 1.0 if face == SOUTH else spec.z - 1.2
                else:
                    pz = spec.z + along
                    px = spec.x + spec.width + 1.0 if face == EAST else spec.x - 1.2
                out.append(Obstacle(px, 0.0, pz, 0.2, levels * 2.0, 0.2, (0.3, 0.3, 0.32), rails=False, name=f"{spec.name}-scaffold-post"))
        return out

    # Building corners, walked "right" along each face (the rail convention: p1 -> p2 is the hanging
    # player's right)
    @staticmethod
    def _face_span(spec, face):
        x0, z0, x1, z1 = spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth
        return {SOUTH: ((x0, z1), (x1, z1)), NORTH: ((x1, z0), (x0, z0)),
                EAST: ((x1, z1), (x1, z0)), WEST: ((x0, z0), (x0, z1))}[face]

    @classmethod
    def course_heights(cls, spec):
        """Heights of the hold courses on a building's walls: the sill and window-head lines of
        every storey (what the facades have), topped up so no gap - the last one to the roof
        edge included - is beyond a climb from the lip below."""
        fh = spec.floor_height
        limit = spec.height - cls.CORNICE - Obstacle.CREVASSE_TOP_CLEARANCE
        ys = sorted(y for n in range(spec.floors) for y in (n * fh + cls.SILL_HEIGHT, n * fh + cls.LINTEL_HEIGHT)
                    if y <= limit + 1e-9)
        out = []
        for y in ys + [spec.height]:
            prev = out[-1] if out else 0.0
            while out and y - prev > config.CLIMB_HOLD_GAP_MAX:
                prev = min(limit, prev + min(config.CLIMB_HOLD_GAP_MAX, 0.5 * (y - prev) + 0.25))
                if prev <= out[-1] + 0.3:
                    break
                out.append(prev)
            if y < spec.height:
                out.append(y)
        return out

    @classmethod
    def _plain_wall_courses(cls, spec, core):
        """Crevasse courses on the plain faces, corner to corner.  A plain face of the core is the
        building side itself and is coplanar with the end faces of the facades beside it, so the
        course runs on over them to the building corner and joins the facade's sill / window-head
        line (or the next plain face's course) there."""
        for face in (SOUTH, NORTH, EAST, WEST):
            if face in spec.facade_faces:
                continue
            a, b = cls._face_span(spec, face)
            for y in cls.course_heights(spec):
                core.add_wall_rail(y, FACE_NORMALS[face], a, b, RailType.LADDER_CREVASSE)

    # A hold can be taken from standing on a platform when it is between these heights above its deck:
    # high enough to hang from with the legs below the deck out of the way (hands >= 0.15 m above the
    # shoulders, see BodyModel.evaluate_grab) and low enough to reach (STAND_REACH).
    PLATFORM_HOLD_MIN = config.SHOULDER_HEIGHT + 0.20
    PLATFORM_HOLD_MAX = config.STAND_REACH - 0.05

    @classmethod
    def _hold_above_platform(cls, spec, core, face, deck_y):
        """A platform that covers the wall (scaffold) blocks the crevasse ladder behind it; make
        sure the wall above its top deck has a hold that can be taken from standing on it, so
        the climb goes on to the roof instead of dead-ending on the scaffold."""
        lo, hi = deck_y + cls.PLATFORM_HOLD_MIN, deck_y + cls.PLATFORM_HOLD_MAX
        if lo <= spec.height <= hi:
            return None                      # the roof edge itself is in reach
        normal = FACE_NORMALS[face]
        existing = [r.height for r in core.rails
                    if abs(float(r.normal[0]) - normal[0]) + abs(float(r.normal[2]) - normal[2]) < 1e-3]
        if any(lo <= y <= hi for y in existing):
            return None
        y = deck_y + 0.5 * (cls.PLATFORM_HOLD_MIN + cls.PLATFORM_HOLD_MAX)
        if y > core.bounds.max[1] - Obstacle.CREVASSE_TOP_CLEARANCE:
            return None                      # (the deck is close enough to the roof to mantle it)
        a, b = cls._face_span(spec, face)
        return core.add_wall_rail(y, normal, a, b, RailType.LADDER_CREVASSE)

    @classmethod
    def _balcony(cls, spec, face, offset, y_top, length=None, depth=None, thick=None, name=None):
        t = cls.BALCONY_THICK if thick is None else thick
        dpt = cls.BALCONY_DEPTH if depth is None else depth
        ln = cls.BALCONY_LENGTH if length is None else length
        name = name or f"{spec.name}-balcony"
        x, z, w, d = spec.x, spec.z, spec.width, spec.depth
        if face == SOUTH:
            return Obstacle(x + offset, y_top - t, z + d, ln, t, dpt, spec.trim_color, add_crevasses=False, name=name)
        if face == NORTH:
            return Obstacle(x + offset, y_top - t, z - dpt, ln, t, dpt, spec.trim_color, add_crevasses=False, name=name)
        if face == EAST:
            return Obstacle(x + w, y_top - t, z + offset, dpt, t, ln, spec.trim_color, add_crevasses=False, name=name)
        return Obstacle(x - dpt, y_top - t, z + offset, dpt, t, ln, spec.trim_color, add_crevasses=False, name=name)


def build_l_block(x, z, name="lblock", color=(0.36, 0.40, 0.44)):
    """Two plain blocks forming an L: convex corners plus one inside corner per hold height."""
    main = Obstacle(x, 0.0, z, 8.0, 12.0, 8.0, color, add_crevasses=True, name=f"{name}-main")
    annex = Obstacle(x - 6.0, 0.0, z + 4.0, 6.0, 6.0, 4.0, color, add_crevasses=True, name=f"{name}-annex")
    return [main, annex]
