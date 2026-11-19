"""The tutorial's waypoint: a glowing yellow arrow floating low in the middle of the view that
turns to point the way to the next place the tutorial wants the player to go.

Drawn in view space (call with an identity modelview, like the held torch), after the world and
with the depth test off, so a wall the player stands against never hides it. It is a solid
arrow - a shaft and a triangular head, both with thickness - tilted towards the camera so it
reads as an arrow lying ahead on the floor, and it bobs gently so the eye finds it.
"""
from __future__ import annotations

import math

from OpenGL.GL import (GL_DEPTH_TEST, GL_NORMALIZE, GL_QUADS, GL_TRIANGLES, glBegin, glDisable, glEnable, glEnd,
                       glNormal3f, glPopMatrix, glPushMatrix, glRotatef, glScalef, glTranslatef, glVertex3f)

ARROW_COLOR = (1.0, 0.86, 0.12)
ARROW_GLOW = (0.75, 0.68, 0.18)       # emission: bright in the dark of the hideout or the street at night
POSITION = (0.0, -0.24, -1.5)         # view space: below the crosshair, centred, clear of the log panel
SCALE = 0.34                          # the arrow below is drawn at this fraction of its metre-scale size
TILT = 38.0                           # degrees the arrow's plane is tipped up towards the camera
BOB = 0.008                           # metres of bob either way

# the arrow in local space: it points along -z; half-thickness t
SHAFT_HALF_W, SHAFT_BACK, SHAFT_FRONT = 0.065, 0.40, -0.02
HEAD_HALF_W, HEAD_BACK, HEAD_TIP = 0.21, -0.02, -0.44
THICK = 0.045


def _quad(a, b, c, d, normal) -> None:
    glNormal3f(*normal)
    for v in (a, b, c, d):
        glVertex3f(*v)


def _prism_sides(top, bottom) -> None:
    """Side walls of a prism from matching rings of ``top`` and ``bottom`` vertices (counter-clockwise seen from above)."""
    n = len(top)
    glBegin(GL_QUADS)
    for i in range(n):
        a, b = top[i], top[(i + 1) % n]
        c, d = bottom[(i + 1) % n], bottom[i]
        ex, ez = b[0] - a[0], b[2] - a[2]
        length = math.hypot(ex, ez) or 1.0
        _quad(a, b, c, d, (ez / length, 0.0, -ex / length))       # outward normal of the edge
    glEnd()


def draw_waypoint_arrow(relative_yaw: float, material, time: float) -> None:
    """Draw the arrow turned by ``relative_yaw`` (degrees clockwise from straight ahead to the
    target) with ``material`` (the scene's shader/fixed-function material setter)."""
    material(color=ARROW_COLOR, emission=ARROW_GLOW, textured=False, fog=0.0)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)                 # the scale below must not shrink the normals (fixed-function lighting)
    glPushMatrix()
    glTranslatef(POSITION[0], POSITION[1] + math.sin(time * 2.6) * BOB, POSITION[2])
    glRotatef(TILT, 1.0, 0.0, 0.0)
    glRotatef(-relative_yaw, 0.0, 1.0, 0.0)
    glScalef(SCALE, SCALE, SCALE)
    t = THICK
    # the shaft: a box
    shaft_top = [(-SHAFT_HALF_W, t, SHAFT_BACK), (SHAFT_HALF_W, t, SHAFT_BACK),
                 (SHAFT_HALF_W, t, SHAFT_FRONT), (-SHAFT_HALF_W, t, SHAFT_FRONT)]
    shaft_bottom = [(x, -t, z) for x, _, z in shaft_top]
    glBegin(GL_QUADS)
    _quad(*shaft_top, (0.0, 1.0, 0.0))
    _quad(*reversed(shaft_bottom), (0.0, -1.0, 0.0))
    glEnd()
    _prism_sides(shaft_top, shaft_bottom)
    # the head: a triangular prism
    head_top = [(-HEAD_HALF_W, t, HEAD_BACK), (HEAD_HALF_W, t, HEAD_BACK), (0.0, t, HEAD_TIP)]
    head_bottom = [(x, -t, z) for x, _, z in head_top]
    glBegin(GL_TRIANGLES)
    glNormal3f(0.0, 1.0, 0.0)
    for v in head_top:
        glVertex3f(*v)
    glNormal3f(0.0, -1.0, 0.0)
    for v in reversed(head_bottom):
        glVertex3f(*v)
    glEnd()
    _prism_sides(head_top, head_bottom)
    glPopMatrix()
    glDisable(GL_NORMALIZE)
    glEnable(GL_DEPTH_TEST)
    material()


def relative_yaw(player, target) -> float:
    """Degrees the player must turn clockwise (as seen from above) to face ``target``, in -180..180."""
    return (player.yaw_towards(target) - player.rotation_y + 180.0) % 360.0 - 180.0
