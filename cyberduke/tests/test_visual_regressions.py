"""Geometry checks on what the renderers draw: nothing clips through anything else, status
lights keep their colour, beams stop on visible surfaces. Everything is drawn into a
context that tracks the model matrix (no OpenGL), so the checks see the world-space boxes
exactly as a backend would."""

import math

import numpy as np
import pytest
from tests.test_render_backends import RecordingContext

from cyberduke.animation import HumanoidPose
from cyberduke.levels import build_facility, build_night_market
from cyberduke.world.level import DoorData


class MatrixRecorder(RecordingContext):
    """Tracks the model matrix like a backend and records every primitive as a world-space
    oriented box: (centre, axes (3x3, columns = the box's scaled local axes), tag)."""

    def __init__(self):
        super().__init__()
        self.m = np.identity(4)
        self.stack = []
        self.tex = None
        self.boxes = []  # (corners (8, 3), tex, kind, local scale)

    def push(self):
        self.stack.append(self.m.copy())

    def pop(self):
        self.m = self.stack.pop()

    def translate(self, x, y, z):
        t = np.identity(4)
        t[:3, 3] = (x, y, z)
        self.m = self.m @ t

    def rotate(self, a, x, y, z):
        from cyberduke.render.backends.modern import rotation

        self.m = self.m @ rotation(a, x, y, z).astype(float)

    def scale(self, x, y, z):
        self.m = self.m @ np.diag((x, y, z, 1.0))

    def _corners(self, lo, hi):
        pts = np.array(
            [(x, y, z, 1.0) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        return (self.m @ pts.T).T[:, :3]

    def cube(self, tex_id=None):
        super().cube(tex_id)
        scale = tuple(np.linalg.norm(self.m[:3, :3], axis=0))
        self.boxes.append(
            (self._corners((-0.5,) * 3, (0.5,) * 3), tex_id, "cube", scale, self.m.copy())
        )

    def shape(self, name, tex_id=None):
        """A rounded shape (``render.shapes``) fills its unit box: recorded as that box,
        its kind the shape's name."""
        scale = tuple(np.linalg.norm(self.m[:3, :3], axis=0))
        self.boxes.append(
            (self._corners((-0.5,) * 3, (0.5,) * 3), tex_id, name, scale, self.m.copy())
        )

    def cylinder(self, r_bot, r_top, h, slices=10, tex_id=None):
        super().cylinder(r_bot, r_top, h, slices, tex_id)
        r = max(r_bot, r_top)
        self.boxes.append(
            (self._corners((-r, 0, -r), (r, h, r)), tex_id, "cylinder", (r, h, r), self.m.copy())
        )


def _inside(point, m):
    """Is a world point inside the unit cube transformed by ``m``?"""
    local = np.linalg.solve(m, np.append(point, 1.0))[:3]
    return bool(np.all(np.abs(local) < 0.5 - 1e-6))


# ----------------------------------------------------------------------------- people
class _Textures(dict):
    """Texture ids by name - the faces given ids as they are asked for (like
    ``render.textures.TextureMap``, which paints them on demand)."""

    def __missing__(self, name):
        if isinstance(name, str) and name.startswith("face|"):
            self[name] = 1000 + len(self)
            return self[name]
        raise KeyError(name)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default


TEXTURES = _Textures({
    "skin": 11,
    "skin_smile": 19,
    "skin_grimace": 20,
    "flesh": 12,
    "shirt": 13,
    "cloth": 14,
    "pants": 15,
    "leather": 16,
    "armor": 17,
    "metal": 18,
})  # fmt: skip


def _draw_person(pose=None, spec=None):
    from cyberduke.ai.guard import GUARD_SPEC
    from cyberduke.render.humanoid import draw_humanoid

    ctx = MatrixRecorder()
    draw_humanoid(ctx, spec or GUARD_SPEC, pose or HumanoidPose(), TEXTURES)
    return ctx


def test_the_face_is_painted_on_the_front_of_the_head_only():
    """The face texture was on the whole head cube: eyes and a mouth on the back of the
    head, on both sides, on the crown and under the chin. It is a skin over the front of
    the (rounded) head only - a hair's breadth proud of it."""
    from cyberduke.ai.guard import GUARD_SPEC
    from cyberduke.render.shapes import mesh

    ctx = _draw_person()
    faces = [b for b in ctx.boxes if b[1] == TEXTURES[GUARD_SPEC.face]]
    assert len(faces) == 1, "one face"
    _, _, kind, _, m = faces[0]
    assert kind == "face"
    v, _ = mesh("face")
    world = v[:, :3].astype(float) @ m[:3, :3].T + m[:3, 3]
    assert world[:, 2].min() > 0.05, "on the front of the head (its face is at z=0.16)"
    assert (v[:, 5] > 0.0).mean() > 0.9, "facing forwards"
    head = next(b for b in ctx.boxes if b[2] == "head")
    proud = np.linalg.norm(m[:3, :3], axis=0) / np.linalg.norm(head[4][:3, :3], axis=0)
    assert np.allclose(proud, proud[0]) and 1.005 < proud[0] < 1.02, "a hair's breadth proud"
    assert np.allclose(m[:3, 3], head[4][:3, 3]), "round the head's middle"
    assert head[1] == TEXTURES["flesh"], "the rest of the head is skin"


def test_faces_wear_expressions_instead_of_a_modelled_smirk():
    """Every character had the same lopsided 'smirk' box stuck over the painted mouth.
    The mouth is painted on the face now (``assets.faces``), in the person's own skin
    tone, with a resting expression - the chef and the stroller smile, everybody else is
    neutral - and the expression of the moment when they talk, eat, smoke or scowl."""
    from cyberduke.ai.guard import ELITE_GUARD_SPEC, GUARD_SPEC
    from cyberduke.simulation import Simulation

    for spec in (GUARD_SPEC, ELITE_GUARD_SPEC):
        assert spec.face == "face|1.00,1.00,1.00||neutral"
        ctx = _draw_person(spec=spec)
        head_front = [b for b in ctx.boxes if b[0][:, 1].min() > 1.75 and b[0][:, 2].max() > 0.16]
        untextured = [b for b in head_front if b[1] is None and b[0][:, 1].max() < 1.95]
        assert not [b for b in untextured if b[0][:, 2].min() > 0.155 and b[3][1] < 0.03], (
            "no lips box on the face"
        )
    street = Simulation(build_night_market())
    people = street.civilians + street.merchants
    moods = {p.role: p.spec.mood for p in people}
    assert moods.pop("sushi_chef") == moods.pop("stroller") == "smile"
    assert set(moods.values()) == {"neutral"}, moods
    assert len({p.spec.face for p in people}) >= len(people) - 2, "their own tones and marks"
    person = next(p for p in street.civilians if p.role == "office_worker")
    ctx = _draw_person(spec=person.spec)
    assert [b for b in ctx.boxes if b[1] == TEXTURES[person.spec.face]], "the resting face"
    talking = HumanoidPose(expression="talk")
    ctx = _draw_person(pose=talking, spec=person.spec)
    assert [b for b in ctx.boxes if b[1] == TEXTURES[person.spec.face_for("talk")]], "talking"
    assert person.spec.face_for("talk").endswith("|talk")


def test_boots_rest_on_the_floor_instead_of_sinking_into_it():
    ctx = _draw_person()
    boots = [b for b in ctx.boxes if b[1] == TEXTURES["leather"] and b[0][:, 1].min() < 0.3]
    assert len(boots) == 2
    for corners, *_ in boots:
        assert corners[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert min(b[0][:, 1].min() for b in ctx.boxes) >= -1e-6, "nothing below the floor"


# A knocked-out body (resting on the floor, never inside anything, as it falls) is covered
# by tests/test_ragdoll.py.


# ----------------------------------------------------------------------------- doors
def _facility_doors():
    return build_facility().doors


@pytest.mark.parametrize("door", _facility_doors(), ids=lambda d: d.door_id)
def test_an_opening_door_slides_into_the_wall_not_through_it(door: DoorData):
    """Leaves are thicker than the walls they slide into: drawn whole, an open blast door was
    painted on both faces of the wall beside the doorway and the maze gate rose out of the
    hedge it slid into. Only what is still in the doorway is drawn."""
    from cyberduke.render.sectors import slid_leaf_piece

    lo, hi = door.obstacle.min_x, door.obstacle.max_x
    for amount in np.linspace(0.0, door.travel, 9):
        for prop in (*door.leaf, *door.indicators):
            piece = slid_leaf_piece(prop, float(amount), lo, hi)
            if piece is None:
                continue
            fp = piece.footprint()
            assert fp.min_x >= lo - 1e-6 and fp.max_x <= hi + 1e-6, (
                door.door_id,
                prop.label,
                amount,
            )
    fully_open = [slid_leaf_piece(p, door.travel, lo, hi) for p in door.leaf]
    assert all(p is None or p.size[0] < 0.15 for p in fully_open), "a fully open door is gone"


def test_beams_see_only_the_part_of_a_door_still_in_the_doorway():
    from cyberduke.render.sectors import door_leaf_boxes
    from cyberduke.simulation import Simulation

    sim = Simulation()
    door = sim.room.doors["blast_door"]
    assert door_leaf_boxes(door) and door_leaf_boxes(door)[0].min_x == pytest.approx(-1.55)
    door.open_amount = door.travel
    assert door_leaf_boxes(door) == []


def test_door_status_lights_show_red_when_locked_and_green_when_open():
    """The strips used their over-bright emission as their diffuse colour too, so every room
    light multiplied them past white: locked and open both rendered plain white."""
    for door in _facility_doors():
        for material, channel in ((door.indicator_closed, 0), (door.indicator_open, 1)):
            assert max(material.color) <= 1.0
            emission = material.emission
            assert emission[channel] >= 1.0
            others = [c for i, c in enumerate(emission) if i != channel]
            assert max(others) <= 0.5, f"{door.door_id}: the hue must survive clamping"


# ----------------------------------------------------------------------------- terminals
@pytest.mark.parametrize("style", ["triple", "double", "single", "damaged", "overgrown"])
def test_desk_chairs_never_pass_through_the_desk(style):
    from cyberduke.entities import Terminal
    from cyberduke.render.entity_renderers import draw_terminal_static

    ctx = MatrixRecorder()
    draw_terminal_static(ctx, Terminal(0.0, 0.0, 0.0, style=style), {})

    def by_scale(*size):
        return [b for b in ctx.boxes if np.allclose(b[3], size, atol=1e-6)]

    desk_top = by_scale(2.55, 0.05, 1.05)[0][4]
    wrist_rest = by_scale(2.45, 0.035, 0.06)[0][4]
    chair = by_scale(0.48, 0.62, 0.06) + by_scale(0.32, 0.14, 0.08) + by_scale(0.07, 0.03, 0.28)
    assert len(chair) == 4, "backrest, headrest and two armrests"
    grid = np.linspace(-0.5, 0.5, 21)
    for *_, m in chair:
        for x in grid:
            for y in grid:
                for z in (-0.5, 0.0, 0.5):
                    p = (m @ np.array((x, y, z, 1.0)))[:3]
                    assert not _inside(p, desk_top) and not _inside(p, wrist_rest), style


# ----------------------------------------------------------------------------- cameras
def test_the_camera_lens_light_sits_in_front_of_its_ring():
    """The glowing sensor used to be drawn inside the lens ring, so the red "seen you" (and
    cyan, and green hijacked) lens was never visible."""
    from cyberduke.entities import SecurityCamera
    from cyberduke.render.entity_renderers import draw_security_camera

    cam = SecurityCamera(0.0, 3.6, 0.0)
    cam.current_tilt = 0.0  # looking straight along +Z
    ctx = MatrixRecorder()
    draw_security_camera(ctx, cam, [])
    ring = [b for b in ctx.boxes if b[2] == "cylinder" and abs(b[3][0] - 0.07) < 1e-6][0]
    led = [b for b in ctx.boxes if np.allclose(b[3], (0.10, 0.10, 0.01), atol=1e-6)][0]
    assert led[0][:, 2].min() >= ring[0][:, 2].max() - 1e-6, "the LED is in front of the ring"
    chassis = [b for b in ctx.boxes if np.allclose(b[3], (0.24, 0.18, 0.42), atol=1e-6)][0]
    assert ring[0][:, 2].min() <= chassis[0][:, 2].max() + 1e-6, "the ring sits on the chassis"


def test_beam_rays_stop_on_what_is_in_their_way_in_3d():
    from cyberduke.core.geometry import AABB
    from cyberduke.render.beams import BoxSet, clip_rays

    low_crate = BoxSet.from_aabbs([AABB(1.0, 2.0, -1, 1, 0.0, 0.5)])
    # from 3 m up, a ray to a point on the floor behind the crate passes over it
    ((p, upright),) = clip_rays((0.0, 3.0, 0.0), [(4.0, 0.02, 0.0)], low_crate)
    assert p == pytest.approx((4.0, 0.02, 0.0))
    # a flat ray runs into its side: an upright face (a drop seam goes down from there)
    ((p, upright),) = clip_rays((0.0, 0.3, 0.0), [(4.0, 0.3, 0.0)], low_crate)
    assert p[0] == pytest.approx(1.0) and upright
    # a steep ray lands on its top: not an upright face
    ((p, upright),) = clip_rays((1.5, 3.0, 0.0), [(1.5, 0.02, 0.0)], low_crate)
    assert p[1] == pytest.approx(0.5) and not upright


def test_beam_rays_hit_rotated_props_exactly():
    from cyberduke.render.beams import BoxSet, clip_rays
    from cyberduke.world.props import Prop

    # a thin wall turned 45 degrees about Y through the origin
    wall = BoxSet.from_props(
        [Prop.box(0.0, 1.0, 0.0, 4.0, 2.0, 0.1, rotate=((45.0, 0.0, 1.0, 0.0),))]
    )
    ((p, upright),) = clip_rays((-2.0, 1.0, 0.0), [(2.0, 1.0, 0.0)], wall)
    # the wall's plane is x = z (through the origin): hit where the ray crosses it
    assert p[0] == pytest.approx(-0.05 / math.sin(math.radians(45.0)), abs=1e-3) and upright


def test_airlock_camera_beam_ends_on_visible_geometry_not_in_mid_air():
    """The fan used to be clipped (in 2D) against the gameplay colliders, which reach 0.3 m
    out from the airlock's pipes: its perimeter and drop seams hung in the air in front of
    the wall. Every ray now ends on the floor or on a drawn surface."""
    from cyberduke.render.beams import BeamOccluders
    from cyberduke.render.entity_renderers import draw_security_camera
    from cyberduke.simulation import Simulation

    sim = Simulation()
    occluders = BeamOccluders(sim.level.props)
    cam = sim.cameras[0]

    class Catch(RecordingContext):
        def __init__(self):
            super().__init__()
            self.segments = []

        def lines(self, segs, rgba, width=1.0):
            self.segments.extend(segs)

    boxes = occluders.around(cam.x, cam.z, cam.max_range)
    lens = (cam.x, cam.y - 0.1, cam.z)
    for t in np.linspace(0.0, 6.0, 13):
        cam.current_yaw = cam.mount_yaw + math.sin(t * cam.sweep_speed) * cam.sweep_angle
        ctx = Catch()
        draw_security_camera(ctx, cam, [], occluders=occluders)
        ends = {seg[1] for seg in ctx.segments if seg[0] == lens}
        assert ends, "the beam was drawn"
        for end in ends:
            if end[1] <= 0.0201:
                continue  # on the floor
            local = np.einsum("nij,nj->ni", boxes.to_local, np.asarray(end) - boxes.centers)
            gap = np.max(np.abs(local) - boxes.halves, axis=1)
            assert gap.min() < 1e-3, f"beam ray ends in mid-air at {end}"


# ----------------------------------------------------------------------------- set dressing
def test_bike_parts_line_up_along_the_bike():
    props = [p for p in build_night_market().props if p.label.startswith(("Bike", "Handlebars"))]
    wheels = [p for p in props if p.label == "Bike wheel"]
    frame = next(p for p in props if p.label == "Bike frame")
    yaw = math.radians(frame.rotate[0][0])
    along = (math.sin(yaw), math.cos(yaw))  # the bike's local +z in the world
    for part in props:
        fp = part.footprint()
        cx, cz = (fp.min_x + fp.max_x) / 2 - frame.x, (fp.min_z + fp.max_z) / 2 - frame.z
        sideways = cx * along[1] - cz * along[0]
        assert abs(sideways) < 0.03, f"{part.label} is {sideways:.2f} m off the bike's line"
    assert len(wheels) == 2
    tail = next(p for p in props if p.label == "Bike tail light")
    bars = next(p for p in props if p.label == "Handlebars")
    to_tail = (tail.x - frame.x) * along[0] + (tail.z - frame.z) * along[1]
    to_bars = (bars.x - frame.x) * along[0] + (bars.z - frame.z) * along[1]
    assert to_tail < 0 < to_bars, "tail light at the back, handlebars at the front"


def test_the_propped_open_dumpster_lid_rises_out_of_the_bin():
    lid = next(p for p in build_night_market().props if p.label == "Dumpster lid, propped open")
    assert lid.footprint().min_y >= 1.3 - 0.05, "the lid's edge no longer dips into the bin"


def test_the_bio_level_sign_sits_under_the_skybridge_deck():
    level = build_facility()
    sign = next(p for p in level.props if p.label.startswith("BIO-LEVEL 3 sign")).footprint()
    deck = next(
        p for p in level.props if p.label.startswith("Rear Transverse Skybridge")
    ).footprint()
    head = next(p for p in level.props if p.label == "Greenhouse door frame head").footprint()
    assert sign.max_y <= deck.min_y, "not into the deck above"
    # not hanging down to the door frame's lower edge: it showed as a sliver in the doorway
    assert sign.min_y >= head.min_y + 0.02


def test_cameras_and_grow_lamps_hang_from_something():
    """Regression: the airlock camera's mounting plate stopped 0.4 m under the ceiling, the
    four greenhouse cameras hung in open air over the maze and the grow lamps had no cables.
    Each camera now has a drop rod up to the surface above it (found once per level); the
    lamps hang from the dome on cables."""
    from cyberduke.levels.facility import gh_dome_glass_y
    from cyberduke.render.beams import BeamOccluders
    from cyberduke.render.entity_renderers import draw_security_camera
    from cyberduke.simulation import Simulation

    sim = Simulation()
    occluders = BeamOccluders(sim.level.props)
    for cam in sim.cameras:
        top = occluders.height_above(cam.x, cam.y + 0.22, cam.z)
        assert top is not None and top > cam.y + 0.22, f"{cam.camera_id}: nothing above it"
        ctx = MatrixRecorder()
        draw_security_camera(ctx, cam, [], mount_top=top)
        highest = max(b[0][:, 1].max() for b in ctx.boxes)
        assert highest == pytest.approx(top, abs=1e-6), f"{cam.camera_id}: rod reaches the mount"
    props = build_facility().props
    lamps = [p for p in props if p.label == "Control node grow lamp housing"]
    cables = [p for p in props if p.label == "Grow lamp cable (to the dome)"]
    assert len(cables) == 2 * len(lamps) == 4
    for c in cables:
        fp = c.footprint()
        assert fp.min_y <= 3.31 and fp.max_y >= gh_dome_glass_y(c.x, c.z) - 0.15


def test_hanging_light_cables_reach_the_ceiling():
    """Regression: the staging bay's cage lights (and the dead one) stopped their cables at
    the default 4.82 m, 18 cm under the 5 m ceiling, in open air between the girders."""
    from cyberduke.render.beams import BeamOccluders

    level = build_facility()
    occluders = BeamOccluders(level.props)
    tops = [(light.x, light.z + dz, light.ceiling_y) for light in level.lights
            if light.fixture == "hanging_cage" for dz in (-0.85, 0.85)]  # fmt: skip
    tops += [
        (p.x, p.z, p.footprint().max_y) for p in level.props if p.label == "Dead cage light cable"
    ]
    assert len(tops) == 11
    for x, z, top in tops:
        above = occluders.height_above(x, top - 0.05, z, limit=3.0)
        assert above is not None and above - top <= 0.025, f"cable at ({x}, {z}) ends in the air"


def test_the_greenhouse_exit_is_the_same_door_the_alley_shows():
    """Leaving the greenhouse and turning round in the alley, it is one door: same yellow
    hazard leaf and size, same frame head, STAFF ONLY sign and red alarm lamp."""
    from cyberduke.levels import build_night_market
    from cyberduke.levels.doorway import STAFF_DOOR, STAFF_DOOR_ALARM

    facility, street = build_facility(), build_night_market()
    exit_door = next(d for d in facility.doors if d.door_id == "street_exit")
    alley_door = next(p for p in street.props if p.label == "The yellow door you came out of")
    (leaf,) = exit_door.leaf
    assert leaf.material == alley_door.material == STAFF_DOOR
    assert leaf.size[:2] == alley_door.size[:2]
    (lamp,) = exit_door.lamps
    assert lamp.material == STAFF_DOOR_ALARM

    def trim(props, z_near):
        return sorted(
            (p.label, p.size, p.material)
            for p in props
            if p.label in ("Door frame head", "HALCYON DYNAMICS - STAFF ONLY")
            and abs(p.z - z_near) < 1.0
        )

    inside = trim(facility.props, leaf.z)
    outside = trim(street.props, alley_door.z)
    assert inside and inside == outside
    assert any(p.label == "Concrete door portal" for p in facility.props), (
        "set in the dome wall, the door keeps the alley's concrete surround"
    )


def test_the_vestibule_ends_at_the_alleys_yellow_door():
    """Past the dome's exit the vestibule used to end in a bare wall with an EXIT sign, so
    the player walked into a wall and found themselves in the alley. Its back wall now
    carries the outer staff door - identical to the one the alley shows - and the exit
    zone is the strip of floor right in front of it."""
    from cyberduke.levels import build_night_market
    from cyberduke.levels.facility import VESTIBULE_DOOR_FACE_Z, VESTIBULE_DOOR_LABEL

    facility, street = build_facility(), build_night_market()
    outer = next(p for p in facility.props if p.label == VESTIBULE_DOOR_LABEL)
    alley_door = next(p for p in street.props if p.label == "The yellow door you came out of")
    assert outer.size == alley_door.size
    assert (outer.material.texture, outer.material.color) == (
        alley_door.material.texture,
        alley_door.material.color,
    )
    # lit by the vestibule's warm strip, not the dome's cyan (under which it went green)
    r, g, b = outer.material.emission
    assert r > g > b > 0.1
    # proud of the back wall, facing back into the vestibule
    assert VESTIBULE_DOOR_FACE_Z - outer.size[2] < outer.z < VESTIBULE_DOOR_FACE_Z

    def near(props, z):
        return sorted(
            (p.label, p.size, p.material)
            for p in props
            if p.label in ("Door frame head", "HALCYON DYNAMICS - STAFF ONLY", "Door alarm light")
            and abs(p.z - z) < 0.3
        )

    assert near(facility.props, outer.z) == near(street.props, alley_door.z)
    # it does not open: the back wall's collider stands right behind it
    from cyberduke.world import Room

    room = Room(facility)
    assert not room.is_valid_position(0.0, VESTIBULE_DOOR_FACE_Z + 0.2)
    zone = facility.exit_zone
    assert zone.max_z == VESTIBULE_DOOR_FACE_Z and zone.max_z - zone.min_z <= 1.0
    assert room.is_valid_position(0.0, (zone.min_z + zone.max_z) / 2.0)
    # nothing lies about in front of the door or in the doorway between the two doors
    clutter = [
        p for p in facility.props if "clutter" in p.tags and 75.9 < p.z < 79.2 and p.size[1] > 0.2
    ]
    assert all(abs(p.x) > 1.0 for p in clutter), "rubbish bags stay by the side walls"


def test_door_lamps_stay_put_and_show_the_lock():
    """The lamp over the staff door is red while it is locked and green once it is
    unlocked - whether the door happens to be open or shut at that moment."""
    from cyberduke.render.sectors import draw_door
    from cyberduke.simulation import Simulation

    sim = Simulation()
    door = sim.room.doors["street_exit"]
    (lamp,) = door.data.lamps
    for unlocked, open_amount in ((False, 0.0), (True, 0.0), (True, door.travel)):
        door.unlocked, door.open_amount = unlocked, open_amount
        ctx = MatrixRecorder()
        draws = []
        ctx.emission = lambda rgb, _d=draws: _d.append(None if rgb is None else tuple(rgb[:3]))
        draw_door(ctx, door, TEXTURES)
        lamp_box = [b for b in ctx.boxes if np.allclose(b[3], lamp.size, atol=1e-6)]
        assert len(lamp_box) == 1, "drawn whether the door is open or not"
        assert np.allclose(lamp_box[0][0].mean(axis=0), (lamp.x, lamp.y, lamp.z), atol=1e-6)
        glow = door.data.indicator_open if unlocked else door.data.indicator_closed
        assert tuple(glow.emission) in draws


def test_the_unlocked_staff_door_waits_shut_until_somebody_walks_up_to_it():
    """Unlocked from the far side of the dome, the exit used to slide fully into its wall
    pocket at once: by the time the netrunner got there, there was no door to see."""
    from cyberduke.simulation import InputState, Simulation

    sim = Simulation()
    door = sim.room.doors["street_exit"]
    door.unlock()
    sim.detection.max_level = 1e9  # nobody here is testing the cameras
    sim.player.x, sim.player.z = 0.0, 66.6  # two cells away
    for _ in range(120):
        sim.update(1 / 60, InputState())
    assert door.unlocked and door.open_amount == 0.0 and not door.passable, "still there, shut"
    sim.player.x, sim.player.z = 0.0, 73.5  # walk up to it
    for _ in range(120):
        sim.update(1 / 60, InputState())
    assert door.fully_open
    sim.player.x, sim.player.z = 0.0, 66.6  # walk away: it shuts behind you
    for _ in range(180):
        sim.update(1 / 60, InputState())
    assert door.open_amount == 0.0 and door.unlocked


def test_what_a_person_holds_is_drawn_in_their_hand():
    """Held things (a takeaway box, the chef's knife) are drawn where the hand is."""
    from cyberduke.animation.reach import hand
    from cyberduke.animation.street import hold_up

    pose = hold_up(2.0)
    bare = _draw_person(pose)
    pose.hand_r = "knife"
    held = _draw_person(pose)
    new = [b for b in held.boxes if not any(np.allclose(b[4], c[4]) for c in bare.boxes)]
    knife = [b for b in new if b[2] == "cube"]  # (the rest: the hand, closing round it)
    assert len(knife) == 3, "handle, bolster and blade"
    moved = {b[2] for b in new if b[2] != "cube"}
    assert moved <= {"palm", "finger", "forearm"}, f"only the hand (and the wrist) moved: {moved}"
    pos, _ = hand(pose, "r")
    for corners, *_ in knife:
        assert np.linalg.norm(corners.mean(axis=0) - pos) < 0.3, "in the hand, not somewhere else"
    handle = knife[0][0].mean(axis=0)
    assert np.linalg.norm(handle - pos) < 0.03, "the handle in the fist"
