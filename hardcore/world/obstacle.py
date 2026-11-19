"""Procedural Obstacle primitive with automatic Ledge Rail extraction."""
import numpy as np
import config
from physics.aabb import AABB
from parkour.ledge_rail import LedgeRail, RailType


# World vertex layout: pos(3) normal(3) uv(2, metres on the face) color(3) extra(3: variant*1000+material*10+role, face width, face height)
VERTEX_FLOATS = 14

# Procedural materials (render/shaders/world.frag draws the detail)
MAT_PLAIN = 0
MAT_CAR_BODY = 1
MAT_CAR_GLASS = 2
MAT_VAN = 3
MAT_BUS = 4
MAT_TRUCK_CAB = 5
MAT_CONTAINER = 6
MAT_WHEEL_SIDE = 7
MAT_WHEEL_TREAD = 8
MAT_CHASSIS = 9
MAT_WALL = 10          # building walls: variant % 3 = plaster / brick / panels, variant // 3 = graffiti / mural / grime
MAT_WINDOW_WALL = 11   # the glass seen through a facade's window recesses
MAT_BILLBOARD = 12     # variant = which advertisement
MAT_LIGHT = 13         # emissive (lamp heads)
MAT_BARK = 14
MAT_LEAVES = 15
MAT_ASPHALT = 16
MAT_PAVEMENT = 17
MAT_TRASH = 18

# Face roles relative to an object's facing direction
ROLE_SIDE, ROLE_FRONT, ROLE_BACK, ROLE_TOP, ROLE_BOTTOM = 0, 1, 2, 3, 4


def material_code(material, role=ROLE_SIDE, variant=0):
    """The per-vertex material word: variant * 1000 + material * 10 + role."""
    return float(int(variant) * 1000 + int(material) * 10 + int(role))


def face_quad(p0, p1, p2, p3, n, color, material=MAT_PLAIN, role=ROLE_SIDE, uvs=None, size=(1.0, 1.0), variant=0):
    """Vertices (4 x VERTEX_FLOATS) and indices of one outward-facing quad (counter-clockwise)."""
    if uvs is None:
        uvs = [(0.0, 0.0), (size[0], 0.0), (size[0], size[1]), (0.0, size[1])]
    verts = []
    code = material_code(material, role, variant)
    for p, uv in zip((p0, p1, p2, p3), uvs):
        verts.append(list(p) + list(n) + list(uv) + list(color) + [code, float(size[0]), float(size[1])])
    return verts, [0, 1, 2, 0, 2, 3]


def box_mesh(centre, axes, half, color, material=MAT_PLAIN, variant=0, facing_axis=None):
    """Decorative box with arbitrary orientation: `axes` are three orthonormal vectors (the box's
    local x, y, z), `half` the half-sizes along them.  Faces get metric uvs; roles are SIDE except
    the +/- faces along `facing_axis` (0..2), which are FRONT / BACK, and local +y, which is TOP."""
    c = np.asarray(centre, dtype=np.float64)
    ax = [np.asarray(a, dtype=np.float64) for a in axes]
    verts, idx = [], []
    for k in range(3):
        for sign in (1.0, -1.0):
            n = ax[k] * sign
            u_axis = ax[(k + 1) % 3] * sign
            v_axis = ax[(k + 2) % 3]
            hu, hv = half[(k + 1) % 3], half[(k + 2) % 3]
            o = c + n * half[k]
            p0 = o - u_axis * hu - v_axis * hv
            p1 = o + u_axis * hu - v_axis * hv
            p2 = o + u_axis * hu + v_axis * hv
            p3 = o - u_axis * hu + v_axis * hv
            role = ROLE_SIDE
            if k == 1:
                role = ROLE_TOP if sign > 0 else ROLE_BOTTOM
            elif facing_axis is not None and k == facing_axis:
                role = ROLE_FRONT if sign > 0 else ROLE_BACK
            fv, fi = face_quad(p0, p1, p2, p3, n, color, material, role, size=(2 * hu, 2 * hv), variant=variant)
            base = len(verts)
            verts.extend(fv)
            idx.extend(i + base for i in fi)
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


class Obstacle:
    def __init__(self, x, y, z, width, height, depth, color=config.COLOR_OBSTACLE, add_crevasses=True,
                 crevasse_faces=None, name="", climb_target=True, rails=True, material=MAT_PLAIN, facing=None,
                 variant=0, face_materials=None):
        """
        crevasse_faces: optional iterable of face normals ((0,0,1), (-1,0,0), ...) that
        receive crevasses; all four faces when None.
        climb_target: False for structural sub-parts whose top is not a place to stand
        (facade bands and piers of a building); used by the climbability tests.
        material / facing / variant: procedural surface detail (vehicles, containers, walls);
        `facing` is the object's forward direction, which decides which face is its front / back /
        sides; `variant` picks a look within a material; `face_materials` maps a face normal
        (as a tuple) to a material for that face alone.
        """
        self.bounds = AABB([x, y, z], [x + width, y + height, z + depth])
        self.color = color
        self.name = name
        self.climb_target = climb_target
        self.material = material
        self.variant = int(variant)
        self.face_materials = {} if face_materials is None else {tuple(int(round(c)) for c in k): v for k, v in face_materials.items()}
        self.facing = None if facing is None else np.asarray(facing, dtype=np.float32)
        self.rails = []
        self.crevasse_faces = None if crevasse_faces is None else [tuple(np.round(np.asarray(n, dtype=float), 3)) for n in crevasse_faces]
        if rails:
            self._generate_rails(add_crevasses)
        self.vertices, self.indices = self._generate_mesh()

    def append_mesh(self, vertices, indices):
        """Attach decorative geometry (wheels, a chassis) that is drawn with the obstacle but has
        no collision and no holds."""
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, VERTEX_FLOATS)
        indices = np.asarray(indices, dtype=np.uint32) + len(self.vertices)
        self.vertices = np.concatenate([self.vertices, vertices], axis=0)
        self.indices = np.concatenate([self.indices, indices], axis=0).astype(np.uint32)

    def __repr__(self):
        return f"Obstacle({self.name or 'block'} min={tuple(self.bounds.min.round(2))} size={tuple(self.bounds.size.round(2))})"

    # Named crevasse heights from the design document
    HEAD_CREVASSE_Y = 2.0
    JUMP_CREVASSE_Y = 3.8
    LADDER_SPACING = 1.8          # additional crevasses on very tall walls
    CREVASSE_TOP_CLEARANCE = 0.3  # a crevasse this close to the top is pointless

    @classmethod
    def crevasse_heights(cls, height):
        """
        Heights (relative to the obstacle base) of the hand/foot crevasses on a
        wall of the given height.  Guarantees that every gap in the resulting
        hold ladder (ground -> crevasses -> top) is climbable by a person
        standing on the previous lip (<= config.CLIMB_HOLD_GAP_MAX).
        """
        heights = []
        limit = height - cls.CREVASSE_TOP_CLEARANCE
        if height <= config.STAND_REACH:
            return heights   # top edge reachable from the ground: no crevasses needed
        for y in (cls.HEAD_CREVASSE_Y, cls.JUMP_CREVASSE_Y):
            if y < limit and (not heights or y - heights[-1] <= config.CLIMB_HOLD_GAP_MAX):
                heights.append(y)
        # Extend the ladder until the top is within reach of the last lip
        while heights and (height - heights[-1]) > config.CLIMB_HOLD_GAP_MAX:
            nxt = heights[-1] + cls.LADDER_SPACING
            if nxt >= limit:
                nxt = limit - 0.01
            if nxt - heights[-1] < 0.5:
                break
            heights.append(nxt)
        return heights

    def _generate_rails(self, add_crevasses):
        b_min = self.bounds.min
        b_max = self.bounds.max
        h = float(b_max[1] - b_min[1])
        y_top = float(b_max[1])

        # (p1 xz, p2 xz, normal) for each vertical face; tangent = hanging player's right
        faces = [
            ((b_min[0], b_max[2]), (b_max[0], b_max[2]), [0.0, 0.0, 1.0]),    # South (+Z)
            ((b_max[0], b_min[2]), (b_min[0], b_min[2]), [0.0, 0.0, -1.0]),   # North (-Z)
            ((b_max[0], b_max[2]), (b_max[0], b_min[2]), [1.0, 0.0, 0.0]),    # East (+X)
            ((b_min[0], b_min[2]), (b_min[0], b_max[2]), [-1.0, 0.0, 0.0]),   # West (-X)
        ]

        def add(y, normal, a, b, rail_type):
            self.rails.append(LedgeRail(
                [a[0], y, a[1]], [b[0], y, b[1]],
                normal=normal, rail_type=rail_type, owner=self
            ))

        # 1. Mantle Rails on all 4 top edges
        for a, b, n in faces:
            add(y_top, n, a, b, RailType.MANTLE_TOP)

        # 2. Crevasses on every exposed vertical face
        if add_crevasses:
            for rel_y in self.crevasse_heights(h):
                y = float(b_min[1]) + rel_y
                if abs(rel_y - self.HEAD_CREVASSE_Y) < 1e-3:
                    rtype = RailType.HEAD_CREVASSE
                elif abs(rel_y - self.JUMP_CREVASSE_Y) < 1e-3:
                    rtype = RailType.JUMP_CREVASSE
                else:
                    rtype = RailType.LADDER_CREVASSE
                for a, b, n in faces:
                    if self.crevasse_faces is not None and tuple(np.round(np.asarray(n, dtype=float), 3)) not in self.crevasse_faces:
                        continue
                    add(y, n, a, b, rtype)

    def add_wall_rail(self, y, normal, a, b, rail_type=RailType.LADDER_CREVASSE):
        """An extra hold across a vertical wall at absolute height `y`, from xz point `a` to `b`
        (a -> b is the hanging player's right, as for the generated rails).  The span may run past
        this box along the same wall plane (a building's wall continues over its facade ends).
        Add holds before the ledge graph is built.  Returns the rail."""
        rail = LedgeRail([a[0], y, a[1]], [b[0], y, b[1]], normal=[float(c) for c in normal],
                         rail_type=rail_type, owner=self)
        self.rails.append(rail)
        return rail

    def _face_role(self, n):
        if n[1] > 0.5:
            return ROLE_TOP
        if n[1] < -0.5:
            return ROLE_BOTTOM
        if self.facing is None:
            return ROLE_SIDE
        d = float(np.dot(n, self.facing))
        if d > 0.5:
            return ROLE_FRONT
        if d < -0.5:
            return ROLE_BACK
        return ROLE_SIDE

    def _generate_mesh(self):
        b_min = self.bounds.min
        b_max = self.bounds.max
        faces = [
            # Front (+Z)
            ([b_min[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_max[2]], [b_max[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_max[2]], [0.0, 0.0, 1.0]),
            # Back (-Z)
            ([b_max[0], b_min[1], b_min[2]], [b_min[0], b_min[1], b_min[2]], [b_min[0], b_max[1], b_min[2]], [b_max[0], b_max[1], b_min[2]], [0.0, 0.0, -1.0]),
            # Top (+Y)
            ([b_min[0], b_max[1], b_max[2]], [b_max[0], b_max[1], b_max[2]], [b_max[0], b_max[1], b_min[2]], [b_min[0], b_max[1], b_min[2]], [0.0, 1.0, 0.0]),
            # Bottom (-Y)
            ([b_min[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_min[2]], [b_max[0], b_min[1], b_max[2]], [b_min[0], b_min[1], b_max[2]], [0.0, -1.0, 0.0]),
            # Right (+X)
            ([b_max[0], b_min[1], b_max[2]], [b_max[0], b_min[1], b_min[2]], [b_max[0], b_max[1], b_min[2]], [b_max[0], b_max[1], b_max[2]], [1.0, 0.0, 0.0]),
            # Left (-X)
            ([b_min[0], b_min[1], b_min[2]], [b_min[0], b_min[1], b_max[2]], [b_min[0], b_max[1], b_max[2]], [b_min[0], b_max[1], b_min[2]], [-1.0, 0.0, 0.0]),
        ]

        verts = []
        indices = []
        for p0, p1, p2, p3, n in faces:
            role = self._face_role(np.asarray(n, dtype=np.float32))
            p0, p1, p2, p3 = (np.asarray(p, dtype=np.float32) for p in (p0, p1, p2, p3))
            w = float(np.linalg.norm(p1 - p0))
            h = float(np.linalg.norm(p3 - p0))
            # metric uv: u along the first edge, v along the second (up on vertical faces).  On the
            # sides of a facing object u runs from its back to its front on both sides, so door and
            # window patterns read the same way on the left and the right.
            uvs = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
            if role == ROLE_SIDE and self.facing is not None and float(np.dot(p1 - p0, self.facing)) < 0.0:
                uvs = [(w, 0.0), (0.0, 0.0), (0.0, h), (w, h)]
            mat = self.face_materials.get(tuple(int(round(c)) for c in n), self.material)
            fv, fi = face_quad(p0, p1, p2, p3, n, self.color, mat, role, uvs, (w, h), variant=self.variant)
            base = len(verts)
            verts.extend(fv)
            indices.extend(i + base for i in fi)

        return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)
