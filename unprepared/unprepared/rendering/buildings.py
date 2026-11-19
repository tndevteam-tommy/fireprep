"""Floors and gabled roofs generated from a level's ``BuildingDef`` footprints."""
from __future__ import annotations

from OpenGL.GL import GL_QUADS, GL_TEXTURE_2D, GL_TRIANGLES, glBegin, glBindTexture, glDisable, glEnable, glEnd, glNormal3f, glTexCoord2f, glVertex3f

from ..core.math3d import Vector3D
from .primitives import draw_textured_box

OVERHANG = 0.4
TILE = 2.0


WALL_THICKNESS = 0.5


def draw_floor(building, texture_id: int) -> None:
    """Floorboards inside the walls (inset by the wall thickness so no edge peeks out at the base)."""
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    # (a cave's floor runs right out to its outline: its walls stand on it, and at its mouth the
    # trail outside meets it with no ground drawn between)
    if getattr(building, "underground", False):
        # (its top alone: the box's sides stood on the outline, in the planes of the walls' outer faces)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        for x, z in ((x1, z1), (x1, z2), (x2, z2), (x2, z1)):
            glTexCoord2f(x / TILE, z / TILE)
            glVertex3f(x, 0.05, z)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        return
    inset = WALL_THICKNESS - 0.02
    draw_textured_box(Vector3D(x1 + inset, 0.0, z1 + inset), Vector3D(x2 - inset, 0.05, z2 - inset), texture_id, tile=TILE)


def draw_ceiling(building, texture_id: int) -> None:
    """The underside of the roof: a flat textured ceiling at wall height, facing down into the
    rooms. It spans the roof's whole base (overhang included) so there is no gap at the eaves."""
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    o = FLAT_ROOF_OVER if getattr(building, "roof_style", "gable") == "flat" else OVERHANG
    if getattr(building, "underground", False):
        o = 0.0                                      # (a cave's roof is the hillside: nothing juts out of its mouth)
    x1, z1, x2, z2 = x1 - o, z1 - o, x2 + o, z2 + o
    y = building.wall_height - 0.02
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    glNormal3f(0.0, -1.0, 0.0)
    glTexCoord2f(0, 0); glVertex3f(x1, y, z1)
    glTexCoord2f((x2 - x1) / TILE, 0); glVertex3f(x2, y, z1)
    glTexCoord2f((x2 - x1) / TILE, (z2 - z1) / TILE); glVertex3f(x2, y, z2)
    glTexCoord2f(0, (z2 - z1) / TILE); glVertex3f(x1, y, z2)
    glEnd()
    glDisable(GL_TEXTURE_2D)


FASCIA = 0.16          # boards around the roof base sealing the seam between roof and ceiling
FASCIA_PROUD = 0.012   # how far the boards stand out from the gable ends


def draw_fascia(building, texture_id: int) -> None:
    """Boards running around the roof base: they close the slit between the roof planes, the
    gables and the ceiling underneath so no light or sky shows through at the eaves."""
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    # a whisker beyond the roof's edge: boards flush with the gable ends would share their plane and z-fight
    out = OVERHANG + FASCIA_PROUD
    x1, z1, x2, z2 = x1 - out, z1 - out, x2 + out, z2 + out
    base = building.wall_height
    lo, hi = base - 0.22, base + 0.04
    draw_textured_box(Vector3D(x1, lo, z1), Vector3D(x2, hi, z1 + FASCIA), texture_id, tile=1.0)
    draw_textured_box(Vector3D(x1, lo, z2 - FASCIA), Vector3D(x2, hi, z2), texture_id, tile=1.0)
    # the end boards fit between the long ones (overlapping at the corners they z-fought)
    draw_textured_box(Vector3D(x1, lo, z1 + FASCIA), Vector3D(x1 + FASCIA, hi, z2 - FASCIA), texture_id, tile=1.0)
    draw_textured_box(Vector3D(x2 - FASCIA, lo, z1 + FASCIA), Vector3D(x2, hi, z2 - FASCIA), texture_id, tile=1.0)


FLAT_ROOF = 0.4           # a flat roof's slab: this thick ...
FLAT_ROOF_OVER = 0.1      # ... and this far out over the walls (faces clear of theirs)
CAVE_SLAB_INSET = 0.01    # a cave's ceiling slab is this far inside its chamber's outline


def draw_flat_roof(building, texture_id: int) -> None:
    """A slab over the walls: a cave's rock ceiling, a keep's leads."""
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    # (a cave's rock ceiling stops a whisker inside its outline: the walls' outer faces stand on the
    # outline, and the slab's sides there shared their planes - at a mouth, for everyone to see)
    o = -CAVE_SLAB_INSET if getattr(building, "underground", False) else FLAT_ROOF_OVER
    base = building.wall_height
    draw_textured_box(Vector3D(x1 - o, base, z1 - o), Vector3D(x2 + o, base + FLAT_ROOF, z2 + o), texture_id, tile=TILE)


SPIRE_OVERHANG = 0.25


def draw_spire(building, texture_id: int, fascia_texture_id: int = None) -> None:
    """A steeple's roof: four steep faces meeting at a point ``roof_height`` over the walls."""
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    o = SPIRE_OVERHANG
    x1, z1, x2, z2 = x1 - o, z1 - o, x2 + o, z2 + o
    base = building.wall_height
    apex = (0.5 * (x1 + x2), base + building.roof_height, 0.5 * (z1 + z2))
    if fascia_texture_id is not None:
        draw_textured_box(Vector3D(x1, base - 0.22, z1), Vector3D(x2, base + 0.02, z2), fascia_texture_id, tile=1.0)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_TRIANGLES)
    half_w = 0.5 * (x2 - x1)
    slant = (half_w * half_w + building.roof_height ** 2) ** 0.5
    for (ax, az), (bx, bz), (nx, nz) in (((x1, z2), (x2, z2), (0.0, 1.0)), ((x2, z2), (x2, z1), (1.0, 0.0)),
                                         ((x2, z1), (x1, z1), (0.0, -1.0)), ((x1, z1), (x1, z2), (-1.0, 0.0))):
        # the face leans back: its normal tilts up by the pitch
        ny = half_w / slant
        k = building.roof_height / slant
        glNormal3f(nx * k, ny, nz * k)
        glTexCoord2f(0.0, 0.0); glVertex3f(ax, base, az)
        glTexCoord2f((x2 - x1) / TILE, 0.0); glVertex3f(bx, base, bz)
        glTexCoord2f((x2 - x1) / TILE / 2.0, slant / TILE); glVertex3f(*apex)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_roof(building, texture_id: int, fascia_texture_id: int = None) -> None:
    """Two sloped planes meeting at a ridge along the longer axis, closed by triangular gables
    and trimmed with fascia boards (or a flat slab, for ``roof_style == "flat"``)."""
    style = getattr(building, "roof_style", "gable")
    if style == "flat":
        draw_flat_roof(building, texture_id)
        return
    if style == "spire":
        draw_spire(building, texture_id, fascia_texture_id)
        return
    if fascia_texture_id is not None:
        draw_fascia(building, fascia_texture_id)
    (x1, z1), (x2, z2) = building.min_xz, building.max_xz
    x1, z1, x2, z2 = x1 - OVERHANG, z1 - OVERHANG, x2 + OVERHANG, z2 + OVERHANG
    base = building.wall_height
    top = base + building.roof_height
    along_x = (x2 - x1) >= (z2 - z1)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    if along_x:
        mid = (z1 + z2) / 2.0
        half = (z2 - z1) / 2.0
        # a slope rises roof_height over the half-depth ``half``, so (0, half, roof_height) is at
        # right angles to it: (0, roof_height, -half) . (0, half, roof_height) == 0
        n = (0.0, half, building.roof_height)
        length = (x2 - x1) / TILE
        slope = (half * half + building.roof_height ** 2) ** 0.5 / TILE
        glBegin(GL_QUADS)
        glNormal3f(0.0, n[1], n[2])                                   # south slope faces +z (and mostly up)
        glTexCoord2f(0, 0); glVertex3f(x1, base, z2)
        glTexCoord2f(length, 0); glVertex3f(x2, base, z2)
        glTexCoord2f(length, slope); glVertex3f(x2, top, mid)
        glTexCoord2f(0, slope); glVertex3f(x1, top, mid)
        glNormal3f(0.0, n[1], -n[2])                                  # north slope faces -z (and mostly up)
        glTexCoord2f(0, 0); glVertex3f(x2, base, z1)
        glTexCoord2f(length, 0); glVertex3f(x1, base, z1)
        glTexCoord2f(length, slope); glVertex3f(x1, top, mid)
        glTexCoord2f(0, slope); glVertex3f(x2, top, mid)
        glEnd()
        glBegin(GL_TRIANGLES)                                         # gable ends
        for gx, nx in ((x1, -1.0), (x2, 1.0)):
            glNormal3f(nx, 0.0, 0.0)
            glTexCoord2f(0, 0); glVertex3f(gx, base, z1)
            glTexCoord2f(1, 0); glVertex3f(gx, base, z2)
            glTexCoord2f(0.5, 1); glVertex3f(gx, top, mid)
        glEnd()
    else:
        mid = (x1 + x2) / 2.0
        half = (x2 - x1) / 2.0
        n = (half, building.roof_height)
        length = (z2 - z1) / TILE
        slope = (half * half + building.roof_height ** 2) ** 0.5 / TILE
        glBegin(GL_QUADS)
        glNormal3f(n[1], n[0], 0.0)                                   # east slope faces +x
        glTexCoord2f(0, 0); glVertex3f(x2, base, z1)
        glTexCoord2f(length, 0); glVertex3f(x2, base, z2)
        glTexCoord2f(length, slope); glVertex3f(mid, top, z2)
        glTexCoord2f(0, slope); glVertex3f(mid, top, z1)
        glNormal3f(-n[1], n[0], 0.0)                                  # west slope faces -x
        glTexCoord2f(0, 0); glVertex3f(x1, base, z2)
        glTexCoord2f(length, 0); glVertex3f(x1, base, z1)
        glTexCoord2f(length, slope); glVertex3f(mid, top, z1)
        glTexCoord2f(0, slope); glVertex3f(mid, top, z2)
        glEnd()
        glBegin(GL_TRIANGLES)
        for gz, nz in ((z1, -1.0), (z2, 1.0)):
            glNormal3f(0.0, 0.0, nz)
            glTexCoord2f(0, 0); glVertex3f(x1, base, gz)
            glTexCoord2f(1, 0); glVertex3f(x2, base, gz)
            glTexCoord2f(0.5, 1); glVertex3f(mid, top, gz)
        glEnd()
    glDisable(GL_TEXTURE_2D)
