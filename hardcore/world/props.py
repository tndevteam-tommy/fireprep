"""Street props, vehicles and natural objects, all built from Obstacle boxes.

Every prop is sized so that its surfaces make sense to a climber: car hoods are
run over (<= 1.0 m), van and bus roofs need a hop and a catch, truck trailers
are reached from the cab, scaffolds are ladders of 2 m platforms, trees are
climbed branch by branch, boulders are stepped terraces.  `axis` is "x" or "z"
(the long axis of the object); `(x, z)` is the centre of the footprint.
"""
import numpy as np
from world.obstacle import (Obstacle, face_quad, box_mesh, MAT_CAR_BODY, MAT_CAR_GLASS, MAT_VAN, MAT_BUS,
                            MAT_TRUCK_CAB, MAT_CONTAINER, MAT_WHEEL_SIDE, MAT_WHEEL_TREAD, MAT_CHASSIS, MAT_BILLBOARD,
                            MAT_LIGHT, MAT_BARK, MAT_LEAVES, MAT_TRASH, MAT_PLAIN, ROLE_SIDE)

COL_CAR = [(0.62, 0.18, 0.16), (0.20, 0.26, 0.48), (0.72, 0.72, 0.74), (0.16, 0.16, 0.17), (0.75, 0.62, 0.20)]
COL_VAN = (0.82, 0.82, 0.80)
COL_VANS = [(0.82, 0.82, 0.80), (0.22, 0.24, 0.30), (0.55, 0.12, 0.12), (0.20, 0.36, 0.52), (0.62, 0.60, 0.55)]
COL_BUS = (0.85, 0.55, 0.12)              # school-bus yellow
COL_BUS_METRO = (0.86, 0.87, 0.88)        # metro buses: white body with a coloured band (variant)
COL_TRUCK = (0.30, 0.42, 0.30)
COL_TRUCKS = [(0.30, 0.42, 0.30), (0.62, 0.14, 0.12), (0.20, 0.28, 0.50), (0.82, 0.80, 0.76), (0.16, 0.16, 0.18), (0.80, 0.52, 0.12)]
COL_TRUCK_BOX = [(0.80, 0.80, 0.78)]     # plus any container colour
COL_METAL = (0.36, 0.38, 0.40)
COL_WOOD = (0.46, 0.32, 0.20)
COL_CONTAINER = [(0.55, 0.20, 0.16), (0.18, 0.35, 0.50), (0.30, 0.45, 0.25), (0.60, 0.50, 0.18)]
COL_TRUNK = (0.36, 0.26, 0.16)
COL_LEAF = (0.18, 0.40, 0.18)
COL_ROCK = (0.44, 0.42, 0.40)
COL_CONCRETE = (0.52, 0.52, 0.50)


COL_TYRE = (0.09, 0.09, 0.10)
COL_FRAME = (0.12, 0.12, 0.13)


def _box(cx, cz, length, width, y0, height, axis, color, facing_sign=0.0, **kw):
    """Axis-aligned box centred at (cx, cz); `length` runs along `axis`.  `facing_sign` (+1 / -1)
    makes the box face along / against its axis (which end is the front)."""
    facing = None
    if facing_sign:
        facing = (facing_sign, 0.0, 0.0) if axis == "x" else (0.0, 0.0, facing_sign)
    if axis == "x":
        return Obstacle(cx - length * 0.5, y0, cz - width * 0.5, length, height, width, color, add_crevasses=False, facing=facing, **kw)
    return Obstacle(cx - width * 0.5, y0, cz - length * 0.5, width, height, length, color, add_crevasses=False, facing=facing, **kw)


def _offset(cx, cz, along, axis):
    return (cx + along, cz) if axis == "x" else (cx, cz + along)


# ------------------------------------------------------ decorative geometry
_WHEEL_CACHE = {}


def _wheel_mesh(cx, cy, cz, radius, width, axis, sides=14):
    """A tyre: a short cylinder whose axle runs across the vehicle (perpendicular to `axis`),
    standing on the ground (cy = radius).  Side faces carry the rim / hub material with a
    -1..1 radial uv; the tread is a dark ring of flat-shaded strips.  Built once per
    (radius, width, axis) and translated into place."""
    key = (round(radius, 4), round(width, 4), axis, sides)
    if key not in _WHEEL_CACHE:
        v, i = _wheel_mesh_at_origin(radius, width, axis, sides)
        _WHEEL_CACHE[key] = (np.array(v, dtype=np.float32), np.array(i, dtype=np.uint32))
    v, i = _WHEEL_CACHE[key]
    v = v.copy()
    v[:, 0] += cx
    v[:, 1] += cy
    v[:, 2] += cz
    return v, i


def _wheel_mesh_at_origin(radius, width, axis, sides):
    verts, idx = [], []
    lateral = np.array([0.0, 0.0, 1.0]) if axis == "x" else np.array([1.0, 0.0, 0.0])
    along = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    c = np.zeros(3, dtype=np.float64)
    half = lateral * (width * 0.5)
    ring = [c + (along * np.cos(2.0 * np.pi * (i + 0.5) / sides) + up * np.sin(2.0 * np.pi * (i + 0.5) / sides)) * radius for i in range(sides)]
    rn = [(along * np.cos(2.0 * np.pi * (i + 0.5) / sides) + up * np.sin(2.0 * np.pi * (i + 0.5) / sides)) for i in range(sides)]

    def add(fv, fi):
        base = len(verts)
        verts.extend(fv)
        idx.extend(i + base for i in fi)

    # tread strips (outward normal = radial)
    for i in range(sides):
        j = (i + 1) % sides
        nrm = (rn[i] + rn[j]) * 0.5
        nrm = nrm / np.linalg.norm(nrm)
        p0, p1, p2, p3 = ring[i] - half, ring[j] - half, ring[j] + half, ring[i] + half
        # wind so the face normal (cross(p1-p0, p3-p0)) points outward
        if np.dot(np.cross(p1 - p0, p3 - p0), nrm) < 0.0:
            p0, p1, p2, p3 = p1, p0, p3, p2
        seg = 2.0 * np.pi * radius / sides
        fv, fi = face_quad(p0, p1, p2, p3, nrm, COL_TYRE, MAT_WHEEL_TREAD, ROLE_SIDE,
                           uvs=[(i * seg, 0.0), ((i + 1) * seg, 0.0), ((i + 1) * seg, width), (i * seg, width)], size=(seg, width))
        add(fv, fi)
    # two side faces: a fan around the centre (uv radial -1..1)
    for sign in (-1.0, 1.0):
        nrm = lateral * sign
        centre = c + half * sign
        for i in range(sides):
            j = (i + 1) % sides
            pa, pb = ring[i] + half * sign, ring[j] + half * sign
            if np.dot(np.cross(pa - centre, pb - centre), nrm) < 0.0:
                pa, pb = pb, pa
            uva = (float(np.dot(pa - centre, along)) / radius, float(np.dot(pa - centre, up)) / radius)
            uvb = (float(np.dot(pb - centre, along)) / radius, float(np.dot(pb - centre, up)) / radius)
            fv, fi = face_quad(centre, pa, pb, pb, nrm, COL_TYRE, MAT_WHEEL_SIDE, ROLE_SIDE,
                               uvs=[(0.0, 0.0), uva, uvb, uvb], size=(1.0, 1.0))
            base = len(verts)
            verts.extend(fv[:3])
            idx.extend([base, base + 1, base + 2])
    return verts, idx


def _deco_box_mesh(cx, cz, length, width, y0, height, axis, color, material=MAT_CHASSIS):
    """Decorative box (no collision): same faces as an Obstacle."""
    tmp = _box(cx, cz, length, width, y0, height, axis, color, rails=False, material=material)
    return tmp.vertices, tmp.indices


WHEEL_JUT = 0.04      # the outer face of a tyre stands this far proud of the vehicle's side


def _add_wheels(body, cx, cz, axis, y_radius, width, axles, half_width, dual=None, facing_sign=1.0):
    """Attach tyres to `body`: one per side on each axle position in `axles` (metres along the
    vehicle from its centre, + toward the front), the outer face WHEEL_JUT proud of the body
    side (`half_width`); `dual` axles get an inner pair too."""
    lateral = (0.0, 1.0) if axis == "x" else (1.0, 0.0)
    for along in axles:
        a = along * facing_sign
        wx, wz = _offset(cx, cz, a, axis)
        tracks = [half_width + WHEEL_JUT - width * 0.5]
        if dual is not None and along in dual:
            tracks.append(half_width + WHEEL_JUT - width * 1.5 - 0.03)
        for t in tracks:
            for side in (-1.0, 1.0):
                px = wx + lateral[0] * side * t
                pz = wz + lateral[1] * side * t
                body.append_mesh(*_wheel_mesh(px, y_radius, pz, y_radius, width, axis))


# ----------------------------------------------------------------- vehicles
def car(cx, cz, axis="z", color=None, name="car", variant=0):
    """Hatchback: body on four tyres (hood / boot top at 1.0 m: run over it), glass cabin.
    variant: 1 = a dented door, 2 = a cracked window, 3 = both."""
    col = color or COL_CAR[0]
    body = _box(cx, cz, 4.4, 1.8, 0.3, 0.7, axis, col, facing_sign=1.0, name=name, material=MAT_CAR_BODY, variant=variant)
    bx, bz = _offset(cx, cz, -0.3, axis)
    cabin = _box(bx, bz, 2.0, 1.5, 1.0, 0.5, axis, col, facing_sign=1.0, name=name + "-cabin", material=MAT_CAR_GLASS, variant=variant)
    _add_wheels(body, cx, cz, axis, 0.33, 0.20, (-1.45, 1.45), 0.9)
    return [body, cabin]


def van(cx, cz, axis="z", name="van", color=None, variant=0):
    """Panel van on four tyres (roof at 2.3: a hop and a catch)."""
    body = _box(cx, cz, 5.2, 2.1, 0.3, 2.0, axis, color or COL_VAN, facing_sign=1.0, name=name, material=MAT_VAN, variant=variant)
    _add_wheels(body, cx, cz, axis, 0.36, 0.22, (-1.65, 1.65), 1.05)
    return [body]


def bus(cx, cz, axis="z", name="bus", style="metro", band=1):
    """City bus (roof at 3.0): single steer tyres, dual rear tyres.  `style` "school" is the
    yellow bus with black rub rails; "metro" is a white body with a coloured band (`band` 1..9
    picks the transit colour)."""
    if style == "school":
        col, variant = COL_BUS, 0
    else:
        col, variant = COL_BUS_METRO, max(1, min(9, int(band)))
    body = _box(cx, cz, 12.0, 2.6, 0.3, 2.7, axis, col, facing_sign=1.0, name=name, material=MAT_BUS, variant=variant)
    _add_wheels(body, cx, cz, axis, 0.50, 0.30, (-3.6, 4.1), 1.3, dual=(-3.6,))
    return [body]


def truck(cx, cz, axis="z", name="truck", cab_color=None, box_color=None, variant=0):
    """Articulated lorry: a cab-over tractor (2 steer tyres, 8 on two drive axles) pulling a
    container on a chassis (8 tyres on two axles) - an 18-wheeler.  The truck faces *against*
    its axis (the cab sits at the negative end, clear of the container with the usual gap
    behind it), so the container doors are at the back."""
    cab_c = _offset(cx, cz, -5.95, axis)
    cab = _box(cab_c[0], cab_c[1], 2.6, 2.5, 0.3, 2.8, axis, cab_color or COL_TRUCK, facing_sign=-1.0, name=name + "-cab",
               material=MAT_TRUCK_CAB, variant=variant)                                             # cab -7.25 .. -4.65, roof at 3.1
    trailer = _box(cx, cz, 8.5, 2.6, 0.9, 3.1, axis, box_color or COL_TRUCK_BOX[0], facing_sign=-1.0, name=name + "-trailer",
                   material=MAT_CONTAINER, variant=variant)                                          # container -4.25 .. 4.25, roof at 4.0
    # tractor frame running back under the container, the container's own chassis rails and
    # the landing-gear legs near its front
    fx, fz = _offset(cx, cz, -3.9, axis)
    cab.append_mesh(*_deco_box_mesh(fx, fz, 6.6, 1.0, 0.55, 0.35, axis, COL_FRAME))
    trailer.append_mesh(*_deco_box_mesh(cx, cz, 8.3, 1.2, 0.62, 0.28, axis, COL_FRAME))
    for side in (-0.9, 0.9):
        lx, lz = _offset(cx, cz, -1.6, axis)
        lx, lz = (lx, lz + side) if axis == "x" else (lx + side, lz)
        trailer.append_mesh(*_deco_box_mesh(lx, lz, 0.12, 0.12, 0.0, 0.62, axis, COL_FRAME))
    _add_wheels(cab, cx, cz, axis, 0.50, 0.30, (-6.6,), 1.25)                            # steer axle
    _add_wheels(cab, cx, cz, axis, 0.50, 0.30, (-3.9, -2.7), 1.25, dual=(-3.9, -2.7))     # drive axles
    _add_wheels(trailer, cx, cz, axis, 0.50, 0.30, (2.2, 3.4), 1.3, dual=(2.2, 3.4))      # trailer axles
    return [cab, trailer]


def dumpster(cx, cz, axis="x", name="dumpster"):
    return [_box(cx, cz, 1.9, 1.3, 0.0, 1.3, axis, (0.20, 0.36, 0.26), name=name)]


def container(cx, cz, y0=0.0, axis="x", color=None, name="container", variant=0):
    """Shipping container sitting on the ground (corrugated, doors at its negative end)."""
    return [_box(cx, cz, 6.1, 2.45, y0, 2.6, axis, color or COL_CONTAINER[0], facing_sign=1.0, name=name,
                 material=MAT_CONTAINER, variant=variant)]


def container_stack(cx, cz, axis="x", name="containers"):
    """Two containers side by side with a third on top straddling them: a 2.6 / 5.2 m step pyramid."""
    w = 2.45
    if axis == "x":
        lo = [(cx, cz - w * 0.5), (cx, cz + w * 0.5)]
    else:
        lo = [(cx - w * 0.5, cz), (cx + w * 0.5, cz)]
    out = container(lo[0][0], lo[0][1], 0.0, axis, COL_CONTAINER[0], name)
    out += container(lo[1][0], lo[1][1], 0.0, axis, COL_CONTAINER[1], name)
    out += container(cx, cz, 2.6, axis, COL_CONTAINER[2], name + "-top")
    return out


# ------------------------------------------------------------- furniture
def bench(cx, cz, axis="x", name="bench"):
    return [_box(cx, cz, 1.8, 0.5, 0.25, 0.25, axis, COL_WOOD, name=name, rails=False)]


def bin_(cx, cz, name="bin"):
    return [Obstacle(cx - 0.3, 0.0, cz - 0.3, 0.6, 0.9, 0.6, COL_METAL, add_crevasses=False, name=name)]


def lamp_post(cx, cz, name="lamp", arm=(1.0, 0.0)):
    """Street light: a 5 m pole with an arm reaching `arm` (unit xz, toward the road) and a lit
    lamp head at its end."""
    post = Obstacle(cx - 0.12, 0.0, cz - 0.12, 0.24, 5.0, 0.24, (0.25, 0.25, 0.27), rails=False, name=name)
    ax, az = float(arm[0]), float(arm[1])
    n = max(1e-6, (ax * ax + az * az) ** 0.5)
    ax, az = ax / n, az / n
    along = np.array([ax, 0.0, az])
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(up, along)
    # arm: a slim bar from the pole top out over the road, then the lamp housing and its lens
    post.append_mesh(*box_mesh((cx + ax * 0.9, 4.92, cz + az * 0.9), (along, up, side), (0.95, 0.06, 0.06), (0.25, 0.25, 0.27)))
    post.append_mesh(*box_mesh((cx + ax * 1.75, 4.86, cz + az * 1.75), (along, up, side), (0.38, 0.10, 0.16), (0.22, 0.22, 0.24)))
    post.append_mesh(*box_mesh((cx + ax * 1.75, 4.745, cz + az * 1.75), (along, up, side), (0.34, 0.035, 0.13), (1.0, 0.93, 0.72),
                               material=MAT_LIGHT))
    return [post]


def bus_shelter(cx, cz, axis="x", name="shelter"):
    """Two posts, a bench and a roof 2.3 m up: hop, catch, top out."""
    out = []
    for along in (-1.8, 1.8):
        px, pz = _offset(cx, cz, along, axis)
        out.append(Obstacle(px - 0.08, 0.0, pz - 0.08, 0.16, 2.3, 0.16, COL_METAL, rails=False, name=name + "-post"))
    out += bench(cx, cz, axis, name + "-bench")
    out.append(_box(cx, cz, 4.0, 1.6, 2.3, 0.12, axis, COL_METAL, name=name + "-roof"))
    return out


def billboard(cx, cz, axis="x", name="billboard", ad=0):
    """A 6 x 3 m panel on two posts, top at 6.5 m (reached from a roof or a truck), carrying
    advertisement `ad` (0..9) on both faces, lit from a small hood."""
    out = []
    for along in (-2.4, 2.4):
        px, pz = _offset(cx, cz, along, axis)
        out.append(Obstacle(px - 0.15, 0.0, pz - 0.15, 0.3, 3.5, 0.3, COL_METAL, rails=False, name=name + "-post"))
    # the panel faces across its axis: both big faces are FRONT / BACK for the ad
    facing = (0.0, 0.0, 1.0) if axis == "x" else (1.0, 0.0, 0.0)
    panel = _box(cx, cz, 6.0, 0.3, 3.5, 3.0, axis, (0.85, 0.85, 0.80), name=name, material=MAT_BILLBOARD, variant=int(ad) % 10)
    panel.facing = np.asarray(facing, dtype=np.float32)
    panel.vertices, panel.indices = panel._generate_mesh()
    # lamps along the top edge
    along_v = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 0.0, 1.0])
    across = np.array(facing, dtype=np.float64)
    for sgn in (-1.0, 1.0):
        for a in (-1.8, 0.0, 1.8):
            c = np.array([cx, 6.62, cz]) + along_v * a + across * sgn * 0.25
            panel.append_mesh(*box_mesh(c, (along_v, np.array([0.0, 1.0, 0.0]), across), (0.22, 0.05, 0.08), (0.22, 0.22, 0.24)))
            panel.append_mesh(*box_mesh(c - np.array([0.0, 0.055, 0.0]), (along_v, np.array([0.0, 1.0, 0.0]), across), (0.18, 0.012, 0.06),
                                        (1.0, 0.95, 0.8), material=MAT_LIGHT))
    out.append(panel)
    return out


def planter(cx, cz, axis="x", name="planter"):
    return [_box(cx, cz, 2.4, 1.0, 0.0, 0.6, axis, COL_CONCRETE, name=name)]


def low_wall(cx, cz, length, axis="x", height=1.2, name="wall"):
    return [_box(cx, cz, length, 0.3, 0.0, height, axis, COL_CONCRETE, name=name)]


# --------------------------------------------------------------- structures
def stairs(cx, cz, axis="z", width=3.0, steps=8, riser=0.2, tread=0.4, direction=1, name="stairs"):
    """Steps rising along `axis` (direction +1/-1); walked up via the step offset."""
    out = []
    for i in range(steps):
        along = direction * (i + 0.5) * tread
        px, pz = _offset(cx, cz, along, axis)
        out.append(_box(px, pz, tread, width, 0.0, riser * (i + 1), axis, COL_CONCRETE, name=name, rails=False))
    return out


def ramp(cx, cz, axis="z", width=4.0, rise=3.0, run=12.0, direction=1, name="ramp"):
    """A stepped vehicle ramp (0.3 m risers) climbing `rise` over `run`; the deck is not included."""
    steps = max(1, int(round(rise / 0.3)))
    tread = run / steps
    out = []
    for i in range(steps):
        along = direction * (i + 0.5) * tread
        px, pz = _offset(cx, cz, along, axis)
        out.append(_box(px, pz, tread, width, 0.0, 0.3 * (i + 1), axis, COL_CONCRETE, name=name, rails=False))
    return out


def parking_deck(x, z, width=14.0, depth=12.0, height=3.0, axis="z", name="deck"):
    """A raised slab on pillars with a 1.0 m parapet and a stepped ramp up one side."""
    out = []
    slab_t = 0.3
    out.append(Obstacle(x, height - slab_t, z, width, slab_t, depth, COL_CONCRETE, add_crevasses=False, name=name))
    for px in (x + 0.5, x + width - 1.0):
        for pz in (z + 0.5, z + depth - 1.0):
            out.append(Obstacle(px, 0.0, pz, 0.5, height - slab_t, 0.5, COL_CONCRETE, rails=False, name=name + "-pillar"))
    # parapet on three sides (the fourth is where the ramp arrives)
    p = 0.25
    out.append(Obstacle(x, height, z, width, 1.0, p, COL_CONCRETE, add_crevasses=False, name=name + "-parapet"))
    out.append(Obstacle(x, height, z + depth - p, width, 1.0, p, COL_CONCRETE, add_crevasses=False, name=name + "-parapet"))
    out.append(Obstacle(x, height, z, p, 1.0, depth, COL_CONCRETE, add_crevasses=False, name=name + "-parapet"))
    # ramp alongside the open (east) side: rises along z over the deck's length and arrives flush
    # with the slab, so the player steps sideways onto the deck at the top
    out += ramp(x + width + 2.0, z, axis="z", width=4.0, rise=height, run=depth, direction=1, name=name + "-ramp")
    return out


def skybridge(x0, z0, x1, z1, y, width=2.0, name="skybridge"):
    """A slab between two points at height y (top surface at y)."""
    xa, xb = min(x0, x1), max(x0, x1)
    za, zb = min(z0, z1), max(z0, z1)
    if xb - xa > zb - za:
        return [Obstacle(xa, y - 0.3, (za + zb) * 0.5 - width * 0.5, xb - xa, 0.3, width, COL_CONCRETE, add_crevasses=False, name=name)]
    return [Obstacle((xa + xb) * 0.5 - width * 0.5, y - 0.3, za, width, 0.3, zb - za, COL_CONCRETE, add_crevasses=False, name=name)]


# ------------------------------------------------------------------ nature
def _rot_y(v, ang):
    c, s_ = np.cos(ang), np.sin(ang)
    return np.array([v[0] * c + v[2] * s_, v[1], -v[0] * s_ + v[2] * c])


def tree(cx, cz, name="tree", seed=0):
    """Trunk, a low cross of branches, a higher cross of branches and a canopy.

    Climb: branch at 2.0 (hang from the ground) -> push up on it -> branch at 3.7 -> canopy top at 5.5.
    The climbable boxes are dressed with bark and leaf materials, a flare at the root, twigs into
    the crown and a cluster of tilted leaf clumps around the canopy box (decorative only), so the
    tree reads as a tree and not a cross of planks.
    """
    rng = np.random.RandomState(int(seed) + 17)
    out = [Obstacle(cx - 0.3, 0.0, cz - 0.3, 0.6, 4.3, 0.6, COL_TRUNK, rails=False, name=name + "-trunk", material=MAT_BARK, variant=seed % 10)]
    trunk = out[0]
    out.append(Obstacle(cx - 2.4, 1.75, cz - 0.15, 4.8, 0.25, 0.3, COL_TRUNK, add_crevasses=False, name=name + "-branch", material=MAT_BARK))
    out.append(Obstacle(cx - 0.15, 1.75, cz - 2.4, 0.3, 0.25, 4.8, COL_TRUNK, add_crevasses=False, name=name + "-branch", material=MAT_BARK))
    out.append(Obstacle(cx - 1.9, 3.45, cz - 0.15, 3.8, 0.25, 0.3, COL_TRUNK, add_crevasses=False, name=name + "-branch", material=MAT_BARK))
    out.append(Obstacle(cx - 0.15, 3.45, cz - 1.9, 0.3, 0.25, 3.8, COL_TRUNK, add_crevasses=False, name=name + "-branch", material=MAT_BARK))
    canopy = Obstacle(cx - 1.2, 4.3, cz - 1.2, 2.4, 1.2, 2.4, COL_LEAF, add_crevasses=False, name=name + "-canopy", material=MAT_LEAVES, variant=seed % 10)
    out.append(canopy)
    up = np.array([0.0, 1.0, 0.0])
    # round off the trunk and the branches: a second box turned 45 deg about each one's axis
    # makes the square timber read as an octagonal limb
    r45 = 1.0 / np.sqrt(2.0)
    trunk.append_mesh(*box_mesh((cx, 2.15, cz), (np.array([r45, 0.0, r45]), up, np.array([-r45, 0.0, r45])),
                                (0.30, 2.15, 0.30), COL_TRUNK, material=MAT_BARK, variant=seed % 10))
    for (c, axis_v, half_len, y) in (((cx, 1.875, cz), np.array([1.0, 0.0, 0.0]), 2.4, 1.875),
                                     ((cx, 1.875, cz), np.array([0.0, 0.0, 1.0]), 2.4, 1.875),
                                     ((cx, 3.575, cz), np.array([1.0, 0.0, 0.0]), 1.9, 3.575),
                                     ((cx, 3.575, cz), np.array([0.0, 0.0, 1.0]), 1.9, 3.575)):
        side = np.cross(up, axis_v)
        e1 = (up + side) * r45
        e2 = (up - side) * r45
        trunk.append_mesh(*box_mesh(c, (axis_v, e1, e2), (half_len, 0.135, 0.135), COL_TRUNK, material=MAT_BARK))
    # root flare: four buttresses around the base
    for k in range(4):
        a = k * np.pi * 0.5 + np.pi * 0.25
        d = _rot_y(np.array([1.0, 0.0, 0.0]), a)
        side = np.cross(up, d)
        trunk.append_mesh(*box_mesh((cx + d[0] * 0.36, 0.22, cz + d[2] * 0.36), (d, up, side), (0.16, 0.22, 0.14), COL_TRUNK, material=MAT_BARK))
    # twigs from the upper branches up into the crown
    for k in range(6):
        a = rng.uniform(0.0, 2.0 * np.pi)
        d = _rot_y(np.array([1.0, 0.0, 0.0]), a)
        tilt = np.array([d[0] * 0.55, 0.83, d[2] * 0.55])
        tilt /= np.linalg.norm(tilt)
        side = np.cross(up, tilt)
        side /= np.linalg.norm(side)
        fwd = np.cross(tilt, side)
        base = np.array([cx + d[0] * 0.9, 3.7, cz + d[2] * 0.9])
        trunk.append_mesh(*box_mesh(base + tilt * 0.6, (side, tilt, fwd), (0.05, 0.65, 0.05), COL_TRUNK, material=MAT_BARK))
    # leaf clumps: tilted boxes of varying size around and above the canopy box
    for k in range(9):
        a = k * 2.0 * np.pi / 9.0 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.9, 1.55)
        d = _rot_y(np.array([1.0, 0.0, 0.0]), a)
        yaw = rng.uniform(0.0, np.pi)
        ez = _rot_y(np.array([0.0, 0.0, 1.0]), yaw)
        tilt = np.array([ez[0] * 0.3, 0.95, ez[2] * 0.3])
        tilt /= np.linalg.norm(tilt)
        ex2 = np.cross(tilt, ez)
        ex2 /= np.linalg.norm(ex2)
        ez2 = np.cross(ex2, tilt)
        half = (rng.uniform(0.55, 0.95), rng.uniform(0.45, 0.7), rng.uniform(0.55, 0.95))
        c = (cx + d[0] * r, 4.3 + rng.uniform(0.2, 1.1), cz + d[2] * r)
        shade = rng.uniform(0.85, 1.15)
        col = (COL_LEAF[0] * shade, COL_LEAF[1] * shade, COL_LEAF[2] * shade)
        canopy.append_mesh(*box_mesh(c, (ex2, tilt, ez2), half, col, material=MAT_LEAVES, variant=(seed + k) % 10))
    # a crown clump on top
    canopy.append_mesh(*box_mesh((cx, 5.75, cz), (_rot_y(np.array([1.0, 0.0, 0.0]), 0.6), up, _rot_y(np.array([0.0, 0.0, 1.0]), 0.6)),
                                 (1.1, 0.55, 1.1), COL_LEAF, material=MAT_LEAVES, variant=(seed + 3) % 10))
    return out


def trash(cx, cz, seed=0, name="trash"):
    """A few bin bags and scraps of litter against a wall (decorative: no collision)."""
    rng = np.random.RandomState(int(seed) + 99)
    host = Obstacle(cx - 0.05, 0.0, cz - 0.05, 0.1, 0.02, 0.1, (0.10, 0.10, 0.11), rails=False, name=name, material=MAT_TRASH)
    up = np.array([0.0, 1.0, 0.0])
    for k in range(int(rng.randint(2, 5))):
        # a bag is a lumpy thing: three overlapping boxes leaning different ways
        bx, bz = cx + rng.uniform(-0.7, 0.7), cz + rng.uniform(-0.5, 0.5)
        for j in range(3):
            a = rng.uniform(0.0, np.pi)
            ex = _rot_y(np.array([1.0, 0.0, 0.0]), a)
            ez = _rot_y(np.array([0.0, 0.0, 1.0]), a)
            lean = rng.uniform(-0.25, 0.25)
            ey = np.array([ex[0] * lean, 1.0, ex[2] * lean])
            ey /= np.linalg.norm(ey)
            ex2 = np.cross(ey, ez)
            ex2 /= np.linalg.norm(ex2)
            ez2 = np.cross(ex2, ey)
            half = (rng.uniform(0.18, 0.32), rng.uniform(0.15, 0.26), rng.uniform(0.16, 0.28))
            c = (bx + rng.uniform(-0.12, 0.12), half[1] * 0.85, bz + rng.uniform(-0.1, 0.1))
            host.append_mesh(*box_mesh(c, (ex2, ey, ez2), half, (0.10, 0.10, 0.12), material=MAT_TRASH, variant=int(rng.randint(10))))
    for k in range(int(rng.randint(3, 8))):
        a = rng.uniform(0.0, np.pi)
        ex = _rot_y(np.array([1.0, 0.0, 0.0]), a)
        ez = _rot_y(np.array([0.0, 0.0, 1.0]), a)
        c = (cx + rng.uniform(-1.6, 1.6), 0.006, cz + rng.uniform(-1.2, 1.2))
        col = [(0.85, 0.82, 0.75), (0.70, 0.55, 0.30), (0.80, 0.20, 0.20), (0.75, 0.75, 0.78)][int(rng.randint(4))]
        host.append_mesh(*box_mesh(c, (ex, up, ez), (rng.uniform(0.08, 0.16), 0.005, rng.uniform(0.06, 0.12)), col, material=MAT_PLAIN))
    return [host]


def boulder(cx, cz, name="boulder"):
    """A stepped rock: three offset slabs no more than 1.2 m apart."""
    return [
        Obstacle(cx - 2.2, 0.0, cz - 1.6, 4.4, 1.1, 3.2, COL_ROCK, add_crevasses=False, name=name),
        Obstacle(cx - 1.4, 1.1, cz - 0.6, 2.8, 1.0, 2.2, (0.48, 0.46, 0.43), add_crevasses=False, name=name + "-mid"),
        Obstacle(cx - 0.5, 2.1, cz + 0.1, 1.4, 0.8, 1.2, (0.52, 0.50, 0.47), add_crevasses=False, name=name + "-top"),
    ]


def terrace_hill(cx, cz, name="hill"):
    """A grassy hill as four 0.9 m terraces."""
    out = []
    for i, (w, d) in enumerate(((14.0, 12.0), (10.5, 9.0), (7.0, 6.0), (3.5, 3.0))):
        out.append(Obstacle(cx - w * 0.5, 0.9 * i, cz - d * 0.5, w, 0.9, d, (0.30 + 0.03 * i, 0.44 + 0.03 * i, 0.26), add_crevasses=False, name=name))
    return out


def fountain(cx, cz, name="fountain"):
    return [
        Obstacle(cx - 3.0, 0.0, cz - 3.0, 6.0, 0.5, 6.0, COL_CONCRETE, add_crevasses=False, name=name),
        Obstacle(cx - 2.0, 0.5, cz - 2.0, 4.0, 0.5, 4.0, (0.30, 0.45, 0.60), add_crevasses=False, name=name + "-basin"),
        Obstacle(cx - 0.5, 1.0, cz - 0.5, 1.0, 2.4, 1.0, COL_CONCRETE, add_crevasses=False, name=name + "-column"),
    ]
