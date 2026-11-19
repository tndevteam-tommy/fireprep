"""Procedural city around the training park.

A street grid (14 m streets: 3 m sidewalks with 0.15 m curbs, 8 m roadway)
divides the map into blocks.  Each block holds 2-4 buildings built from the
building template (random storeys, facades toward the streets, doors, awnings,
balconies, fire escapes, scaffolding) separated by alleys in cat-leap range or
flush so their ledges join; some lots become set-pieces instead (a parking deck
with a stepped ramp, a container yard, a construction site).  Streets carry
parked cars, vans, buses and trucks; sidewalks get benches, bins, planters,
bus shelters, lamp posts and billboards; the park gets trees, boulders, a
fountain and a terraced hill.  Everything is deterministic for a given seed.
"""
import numpy as np
import config
from world.obstacle import Obstacle, MAT_ASPHALT, MAT_PAVEMENT
from world.building import (BuildingGenerator, BuildingSpec, SOUTH, NORTH, EAST, WEST,
                            WALL_PLASTER, WALL_BRICK, WALL_PANELS, DECOR_NONE, DECOR_GRAFFITI, DECOR_MURAL, DECOR_GRIME)
from world import props


class Rect:
    __slots__ = ("x0", "z0", "x1", "z1")

    def __init__(self, x0, z0, x1, z1):
        self.x0, self.z0, self.x1, self.z1 = float(x0), float(z0), float(x1), float(z1)

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def d(self):
        return self.z1 - self.z0

    @property
    def cx(self):
        return (self.x0 + self.x1) * 0.5

    @property
    def cz(self):
        return (self.z0 + self.z1) * 0.5

    def overlaps(self, o, pad=0.0):
        return not (self.x1 + pad <= o.x0 or o.x1 + pad <= self.x0 or self.z1 + pad <= o.z0 or o.z1 + pad <= self.z0)

    def __repr__(self):
        return f"Rect({self.x0:.0f},{self.z0:.0f},{self.x1:.0f},{self.z1:.0f})"


def footprint(obstacles):
    xs0 = min(float(o.bounds.min[0]) for o in obstacles)
    zs0 = min(float(o.bounds.min[2]) for o in obstacles)
    xs1 = max(float(o.bounds.max[0]) for o in obstacles)
    zs1 = max(float(o.bounds.max[2]) for o in obstacles)
    return Rect(xs0, zs0, xs1, zs1)


class CityGenerator:
    STREET = 14.0
    SIDEWALK = 3.0
    CURB = 0.15
    # Block edges (metres). The training park occupies x in [-33, 33], z in [-33, 53].
    X_BLOCKS = [(-141.0, -101.0), (-87.0, -47.0), (-33.0, 33.0), (47.0, 87.0), (101.0, 141.0)]
    Z_BLOCKS = [(-141.0, -101.0), (-87.0, -47.0), (-33.0, 7.0), (21.0, 53.0), (67.0, 107.0), (121.0, 161.0)]
    PARK = Rect(-33.0, -33.0, 33.0, 53.0)
    EXTENT = Rect(-150.0, -150.0, 150.0, 170.0)
    WALL_HEIGHT = 12.0

    def __init__(self, seed=7):
        self.rng = np.random.RandomState(seed)
        self.obstacles = []
        self.occupied = []          # Rect footprints of placed things (collision avoidance)
        self.buildings = []         # (spec, parts)
        self.protrusions = []       # building parts sticking out over the sidewalk (awnings, balconies, ...)
        self.stats = {"buildings": 0, "vehicles": 0, "props": 0, "setpieces": 0}

    # --------------------------------------------------------------- helpers
    def _place(self, parts, rect=None, pad=0.0):
        r = rect or footprint(parts)
        if any(r.overlaps(o, pad) for o in self.occupied):
            return False
        self.occupied.append(r)
        self.obstacles.extend(parts)
        return True

    def _rand(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def _choice(self, seq):
        return seq[int(self.rng.randint(len(seq)))]

    # ----------------------------------------------------------------- build
    def build(self, park_obstacles):
        ext = self.EXTENT
        # Ground and perimeter (climbable 12 m walls with a crevasse ladder)
        ground = Obstacle(ext.x0 - 5, -0.5, ext.z0 - 5, ext.w + 10, 0.5, ext.d + 10, color=config.COLOR_FLOOR, add_crevasses=False, rails=False,
                          material=MAT_ASPHALT)
        self.obstacles.append(ground)
        wall_col = (0.18, 0.19, 0.22)
        t = 1.0
        self.obstacles.append(Obstacle(ext.x0 - t, 0.0, ext.z0 - t, t, self.WALL_HEIGHT, ext.d + 2 * t, wall_col, name="city-wall"))
        self.obstacles.append(Obstacle(ext.x1, 0.0, ext.z0 - t, t, self.WALL_HEIGHT, ext.d + 2 * t, wall_col, name="city-wall"))
        self.obstacles.append(Obstacle(ext.x0 - t, 0.0, ext.z0 - t, ext.w + 2 * t, self.WALL_HEIGHT, t, wall_col, name="city-wall"))
        self.obstacles.append(Obstacle(ext.x0 - t, 0.0, ext.z1, ext.w + 2 * t, self.WALL_HEIGHT, t, wall_col, name="city-wall"))

        # The training park and its own decoration
        self.obstacles.extend(park_obstacles)
        self.occupied.append(Rect(self.PARK.x0, self.PARK.z0, self.PARK.x1, self.PARK.z1))
        self._park_nature()

        # Blocks
        blocks = []
        for xa, xb in self.X_BLOCKS:
            for za, zb in self.Z_BLOCKS:
                r = Rect(xa, za, xb, zb)
                if r.overlaps(self.PARK, pad=-1.0):
                    continue
                blocks.append(r)
        setpiece_blocks = {1: "deck", 7: "containers", 13: "construction", 19: "deck", 23: "containers"}
        for i, block in enumerate(blocks):
            self._sidewalks(block)
            kind = setpiece_blocks.get(i)
            if kind == "deck":
                self._parking_block(block)
            elif kind == "containers":
                self._container_yard(block)
            elif kind == "construction":
                self._construction_site(block)
            else:
                self._building_block(block)
            self._street_furniture(block)
        self._vehicles(blocks)
        return self.obstacles

    # --------------------------------------------------------------- streets
    def _sidewalks(self, block):
        s, c = self.SIDEWALK, self.CURB
        col = (0.50, 0.50, 0.48)
        kw = dict(rails=False, name="sidewalk", material=MAT_PAVEMENT)
        pieces = [
            Obstacle(block.x0 - s, 0.0, block.z0 - s, block.w + 2 * s, c, s, col, **kw),
            Obstacle(block.x0 - s, 0.0, block.z1, block.w + 2 * s, c, s, col, **kw),
            Obstacle(block.x0 - s, 0.0, block.z0, s, c, block.d, col, **kw),
            Obstacle(block.x1, 0.0, block.z0, s, c, block.d, col, **kw),
        ]
        self.obstacles.extend(pieces)

    def _vehicles(self, blocks):
        """Parked vehicles along the curbs of every street, nose-to-tail with gaps."""
        lanes = []
        for b in blocks:
            # lanes run along each side of the block, just off the curb
            lanes.append(("z", b.x0 - self.SIDEWALK - 1.3, b.z0 - 2.0, b.z1 + 2.0))
            lanes.append(("z", b.x1 + self.SIDEWALK + 1.3, b.z0 - 2.0, b.z1 + 2.0))
            lanes.append(("x", b.z0 - self.SIDEWALK - 1.3, b.x0 - 2.0, b.x1 + 2.0))
            lanes.append(("x", b.z1 + self.SIDEWALK + 1.3, b.x0 - 2.0, b.x1 + 2.0))
        kinds = [props.car] * 6 + [props.van] * 2 + [props.bus, props.truck]
        for axis, fixed, a0, a1 in lanes:
            pos = a0 + self._rand(0.0, 6.0)
            while pos < a1 - 5.0:
                kind = self._choice(kinds)
                length = {props.car: 4.4, props.van: 5.2, props.bus: 12.0, props.truck: 13.0}[kind]
                if pos + length > a1:
                    break
                centre = pos + length * 0.5 + (2.0 if kind is props.truck else 0.0)
                cx, cz = (centre, fixed) if axis == "x" else (fixed, centre)
                dent = int(self.rng.randint(4)) if self.rng.rand() < 0.35 else 0      # the odd dent / cracked window
                if kind is props.car:
                    parts = props.car(cx, cz, axis, color=self._choice(props.COL_CAR), variant=dent)
                elif kind is props.van:
                    parts = props.van(cx, cz, axis, color=self._choice(props.COL_VANS), variant=dent)
                elif kind is props.bus:
                    # mostly metro buses in the transit colours, the odd yellow school bus
                    if self.rng.rand() < 0.2:
                        parts = props.bus(cx, cz, axis, style="school")
                    else:
                        parts = props.bus(cx, cz, axis, style="metro", band=int(self.rng.randint(1, 10)))
                else:
                    box_col = self._choice(props.COL_CONTAINER + props.COL_TRUCK_BOX)
                    parts = props.truck(cx, cz, axis, cab_color=self._choice(props.COL_TRUCKS), box_color=box_col, variant=dent)
                if self._place(parts, pad=0.5):
                    self.stats["vehicles"] += 1
                pos += length + self._rand(1.5, 9.0)

    def _street_furniture(self, block):
        """Benches, bins, planters, lamp posts, a shelter and the odd billboard on the sidewalks."""
        s = self.SIDEWALK
        sides = [
            ("x", block.z0 - s * 0.5, block.x0, block.x1),
            ("x", block.z1 + s * 0.5, block.x0, block.x1),
            ("z", block.x0 - s * 0.5, block.z0, block.z1),
            ("z", block.x1 + s * 0.5, block.z0, block.z1),
        ]
        for axis, fixed, a0, a1 in sides:
            # which way is the road from this sidewalk (lamp arms reach out over it)
            if axis == "x":
                road = (0.0, -1.0) if fixed < block.cz else (0.0, 1.0)
            else:
                road = (-1.0, 0.0) if fixed < block.cx else (1.0, 0.0)
            pos = a0 + self._rand(2.0, 8.0)
            while pos < a1 - 4.0:
                roll = self.rng.rand()
                cx, cz = (pos, fixed) if axis == "x" else (fixed, pos)
                if roll < 0.26:
                    parts = props.bench(cx, cz, axis)
                elif roll < 0.40:
                    parts = props.bin_(cx, cz)
                elif roll < 0.54:
                    parts = props.planter(cx, cz, axis)
                elif roll < 0.72:
                    parts = props.lamp_post(cx, cz, arm=road)
                elif roll < 0.84:
                    parts = props.bus_shelter(cx, cz, axis)
                elif roll < 0.92:
                    # litter piles up against the building side of the sidewalk
                    tx, tz = (cx - road[0] * 1.0, cz - road[1] * 1.0)
                    parts = props.trash(tx, tz, seed=int(self.rng.randint(1000)))
                else:
                    parts = props.billboard(cx, cz, axis, ad=int(self.rng.randint(10)))
                if not all(self._clear_of(p, self.protrusions, clearance=0.1) for p in parts):
                    pass                            # would stand in an awning / balcony / scaffold
                elif self._place(parts, pad=0.3):
                    self.stats["props"] += 1
                pos += self._rand(7.0, 14.0)

    # -------------------------------------------------------------- buildings
    def _split(self, total, gaps=(0.0, 2.8, 3.2, 3.8, 4.5)):
        """Two building extents plus an alley (0 = flush) filling `total` metres."""
        gap = self._choice(gaps)
        a = self._rand(9.0, 16.0)
        b = total - a - gap
        if b < 8.0:
            b = 8.0
            a = total - gap - b
        return a, gap, b

    def _building_block(self, block):
        """2 columns x 1-2 rows of buildings; neighbours are flush or separated by cat-leap alleys."""
        rows = 2 if block.d > 30 else 1
        wa, gx, wb = self._split(block.w)
        if rows == 2:
            da, gz, db = self._split(block.d)
        else:
            da, gz, db = min(block.d, self._rand(9.0, 16.0)), 0.0, 0.0
        placed = []
        for i in range(2):
            for j in range(rows):
                w = wa if i == 0 else wb
                d = da if j == 0 else db
                x = block.x0 if i == 0 else block.x0 + wa + gx
                z = block.z0 if j == 0 else block.z0 + da + gz
                facades = [WEST if i == 0 else EAST]
                if rows == 1 or j == 0:
                    facades.append(NORTH)
                if rows == 1 or j == rows - 1:
                    facades.append(SOUTH)
                if self.rng.rand() < 0.4 and len(facades) > 1:
                    facades = facades[:1]
                plain = [f for f in (SOUTH, NORTH, EAST, WEST) if f not in facades]
                floors = int(self.rng.randint(2, 7))
                balconies = []
                fire_escape = None
                scaffold = None
                if plain:
                    face = self._choice(plain)
                    length = w if face in (SOUTH, NORTH) else d
                    if self.rng.rand() < 0.5:
                        for fl in range(1, floors):
                            if self.rng.rand() < 0.6:
                                balconies.append((face, fl, self._rand(0.5, max(0.6, length - 3.2))))
                    elif floors >= 3:
                        fire_escape = (face, self._rand(0.5, max(0.6, length - 2.5)))
                    if self.rng.rand() < 0.25 and len(plain) > 1:
                        f2 = [f for f in plain if f != face][0]
                        l2 = w if f2 in (SOUTH, NORTH) else d
                        scaffold = (f2, 0.5, max(3.0, l2 - 1.0), min(3, floors))
                # a worn city: plaster, brick and panel walls in many colours, some with graffiti
                # at street level, a mural on a blind wall, or heavy water damage
                style = self._choice([WALL_PLASTER, WALL_PLASTER, WALL_BRICK, WALL_BRICK, WALL_PANELS])
                palettes = {
                    WALL_PLASTER: [((0.62, 0.58, 0.50), (0.66, 0.62, 0.54), (0.74, 0.71, 0.65)),
                                   ((0.55, 0.62, 0.64), (0.58, 0.64, 0.66), (0.70, 0.74, 0.76)),
                                   ((0.66, 0.55, 0.40), (0.70, 0.58, 0.42), (0.78, 0.70, 0.58)),
                                   ((0.48, 0.56, 0.46), (0.52, 0.58, 0.48), (0.66, 0.70, 0.62)),
                                   ((0.70, 0.66, 0.60), (0.72, 0.68, 0.62), (0.80, 0.78, 0.74)),
                                   ((0.58, 0.42, 0.40), (0.60, 0.44, 0.42), (0.72, 0.62, 0.58))],
                    WALL_BRICK: [((0.56, 0.30, 0.24), (0.58, 0.32, 0.26), (0.68, 0.60, 0.52)),
                                 ((0.48, 0.26, 0.22), (0.50, 0.28, 0.24), (0.62, 0.56, 0.50)),
                                 ((0.60, 0.40, 0.30), (0.62, 0.42, 0.32), (0.70, 0.64, 0.56)),
                                 ((0.66, 0.56, 0.44), (0.66, 0.56, 0.44), (0.74, 0.70, 0.62))],
                    WALL_PANELS: [((0.46, 0.48, 0.50), (0.50, 0.52, 0.54), (0.62, 0.64, 0.68)),
                                  ((0.36, 0.40, 0.46), (0.40, 0.44, 0.50), (0.58, 0.60, 0.64)),
                                  ((0.52, 0.50, 0.46), (0.56, 0.54, 0.50), (0.66, 0.64, 0.60)),
                                  ((0.40, 0.46, 0.42), (0.44, 0.50, 0.46), (0.60, 0.64, 0.60))],
                }
                col = self._choice(palettes[style])
                decor = self._choice([DECOR_NONE, DECOR_NONE, DECOR_GRAFFITI, DECOR_GRAFFITI, DECOR_MURAL, DECOR_GRIME, DECOR_GRIME])
                if decor == DECOR_MURAL and not plain:
                    decor = DECOR_GRAFFITI
                spec = BuildingSpec(x, z, w, d, floors, facade_faces=tuple(facades), door_face=facades[0],
                                    balconies=tuple(balconies), rooftop=self.rng.rand() < 0.7,
                                    name=f"bld-{len(self.buildings)}", color=col[0], facade_color=col[1], trim_color=col[2],
                                    awning=self.rng.rand() < 0.5, fire_escape=fire_escape, scaffold=scaffold,
                                    style=style, decor=decor)
                parts = BuildingGenerator.build(spec)
                # footprint check against everything except the other buildings of this block (alleys are intended)
                if self._place(parts, Rect(x - 0.3, z - 0.3, x + w + 0.3, z + d + 0.3)):
                    self.buildings.append((spec, parts))
                    self.stats["buildings"] += 1
                    placed.append((spec, parts))
        self._skybridges(block)
        for spec, parts in placed:
            self._note_protrusions(spec, parts)

    def _note_protrusions(self, spec, parts):
        """Awnings, balconies, fire escapes and scaffolds stick out over the sidewalk; street
        furniture is kept out of them in 3D (a bench under an awning is fine, a billboard post
        through it is not)."""
        x0, z0, x1, z1 = spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth
        for p in parts:
            b = p.bounds
            if b.min[0] < x0 - 1e-3 or b.min[2] < z0 - 1e-3 or b.max[0] > x1 + 1e-3 or b.max[2] > z1 + 1e-3:
                if not p.name.endswith("cornice"):          # the cornice crowns the roof, far above
                    self.protrusions.append(p)

    @staticmethod
    def _clear_of(box, obstacles, clearance=0.05):
        """True if `box` (an Obstacle) keeps `clearance` from every obstacle in the list."""
        lo, hi = box.bounds.min - clearance, box.bounds.max + clearance
        for o in obstacles:
            b = o.bounds
            if (b.max[0] > lo[0] and b.min[0] < hi[0] and b.max[1] > lo[1] and b.min[1] < hi[1]
                    and b.max[2] > lo[2] and b.min[2] < hi[2]):
                return False
        return True

    def _skybridges(self, block):
        """Join two buildings of this block that face each other across an alley at a shared floor."""
        specs = [s for s, _ in self.buildings if block.x0 - 1 <= s.x <= block.x1 and block.z0 - 1 <= s.z <= block.z1]
        for a in specs:
            for b in specs:
                if a is b or a.floors < 3 or b.floors < 3:
                    continue
                ax1, bx0 = a.x + a.width, b.x
                gap = bx0 - ax1
                if 2.5 <= gap <= 5.0:
                    z_lo, z_hi = max(a.z, b.z), min(a.z + a.depth, b.z + b.depth)
                    if z_hi - z_lo >= 3.0 and self.rng.rand() < 0.6:
                        level = min(a.floors, b.floors) - 1
                        y = level * 3.0 + 0.3
                        # the middle of the shared wall if it is clear, else the nearest clear spot: never
                        # through a balcony / fire-escape landing / scaffold on either alley wall
                        mid = (z_lo + z_hi) * 0.5
                        # whatever stands in the alley between the two walls (the walls themselves are
                        # where the bridge is supposed to rest)
                        near = [o for o in self.obstacles
                                if o.bounds.max[0] > ax1 + 0.01 and o.bounds.min[0] < bx0 - 0.01
                                and o.bounds.max[2] > z_lo - 0.5 and o.bounds.min[2] < z_hi + 0.5
                                and o.bounds.max[1] > y - 1.0 and o.bounds.min[1] < y + 1.0]
                        offsets = [0.0] + [sgn * k * 0.5 for k in range(1, 40) for sgn in (1.0, -1.0)]
                        for off in offsets:
                            zc = mid + off
                            if zc - 1.0 < z_lo + 0.3 or zc + 1.0 > z_hi - 0.3:
                                continue
                            bridge = props.skybridge(ax1, zc, bx0, zc, y, name="skybridge")
                            if all(self._clear_of(p, near) for p in bridge):
                                self.obstacles.extend(bridge)
                                self.stats["setpieces"] += 1
                                return

    # -------------------------------------------------------------- setpieces
    def _parking_block(self, block):
        deck = props.parking_deck(block.x0 + 2.0, block.z0 + 2.0, width=16.0, depth=14.0, height=3.0)
        self._place(deck, pad=0.5)
        # a second, taller deck reached from the first via a container step
        deck2 = props.parking_deck(block.x0 + 2.0, block.z1 - 16.0, width=16.0, depth=14.0, height=5.6)
        self._place(deck2, pad=0.5)
        self._place(props.container_stack(block.x1 - 8.0, block.cz, axis="z"), pad=0.5)
        self.stats["setpieces"] += 1

    def _container_yard(self, block):
        for i in range(3):
            for j in range(2):
                cx = block.x0 + 6.0 + i * 12.0
                cz = block.z0 + 5.0 + j * 8.0
                if self.rng.rand() < 0.5:
                    parts = props.container_stack(cx, cz, axis="x")
                else:
                    parts = props.container(cx, cz, 0.0, "x", self._choice(props.COL_CONTAINER))
                self._place(parts, pad=0.5)
        self._place(props.truck(block.x1 - 4.0, block.cz, axis="z"), pad=0.5)
        self.stats["setpieces"] += 1

    def _construction_site(self, block):
        """An unfinished concrete frame: slabs on pillars every 3 m with scaffolding and a crane base."""
        x, z, w, d = block.x0 + 3.0, block.z0 + 3.0, 18.0, 14.0
        parts = []
        for lv in range(1, 5):
            parts.append(Obstacle(x, lv * 3.0 - 0.3, z, w, 0.3, d, props.COL_CONCRETE, add_crevasses=False, name="frame-slab"))
        for px in np.linspace(x, x + w - 0.5, 4):
            for pz in np.linspace(z, z + d - 0.5, 3):
                parts.append(Obstacle(float(px), 0.0, float(pz), 0.5, 12.0, 0.5, props.COL_CONCRETE, rails=False, name="frame-pillar"))
        # scaffold towers on the south face of the frame
        for lv in range(1, 6):
            parts.append(Obstacle(x + 2.0, lv * 2.0 - 0.1, z + d, 6.0, 0.1, 1.2, props.COL_METAL, add_crevasses=False, name="scaffold"))
        for sx in (x + 2.1, x + 7.7):
            parts.append(Obstacle(sx, 0.0, z + d + 1.0, 0.2, 10.0, 0.2, props.COL_METAL, rails=False, name="scaffold-post"))
        self._place(parts, pad=0.5)
        self._place(props.dumpster(block.x1 - 4.0, block.z0 + 4.0), pad=0.5)
        self._place(props.container(block.x1 - 6.0, block.z1 - 4.0, 0.0, "x", props.COL_CONTAINER[3]), pad=0.5)
        self.stats["setpieces"] += 1

    # ------------------------------------------------------------------ park
    def _park_nature(self):
        items = [
            props.tree(-25.0, -25.0, seed=1), props.tree(25.0, -25.0, seed=2), props.tree(27.0, 28.0, seed=3), props.tree(-12.0, 50.5, seed=4),
            props.boulder(22.0, 18.0), props.fountain(22.0, 42.0), props.terrace_hill(21.0, -14.0),
            props.bench(-14.0, -26.0, "x"), props.bench(14.0, -26.0, "x"), props.planter(0.0, -28.0, "x"),
            props.bench(27.0, 46.0, "z"),
        ]
        for parts in items:
            self.obstacles.extend(parts)
            self.stats["props"] += 1
