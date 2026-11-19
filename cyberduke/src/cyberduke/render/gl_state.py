"""Small context managers that make fixed-function GL state changes exception-safe and
remove the push/pop and emission-on/emission-off boilerplate from draw code."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from OpenGL.GL import (
    GL_EMISSION,
    GL_FRONT,
    GL_FRONT_AND_BACK,
    GL_LIGHTING,
    GL_SHININESS,
    GL_SPECULAR,
    glColor3f,
    glDisable,
    glEnable,
    glMaterialf,
    glMaterialfv,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
)

NO_EMISSION = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)


@contextmanager
def pushed_matrix() -> Iterator[None]:
    glPushMatrix()
    try:
        yield
    finally:
        glPopMatrix()


@contextmanager
def transformed(
    translate: Sequence[float] | None = None,
    rotate: Sequence[tuple[float, float, float, float]] | None = None,
    scale: Sequence[float] | None = None,
) -> Iterator[None]:
    """push; translate; rotate (each ``(angle, x, y, z)`` in order); scale; ...; pop."""
    glPushMatrix()
    try:
        if translate is not None:
            glTranslatef(*translate)
        for angle, ax, ay, az in rotate or ():
            glRotatef(angle, ax, ay, az)
        if scale is not None:
            glScalef(*scale)
        yield
    finally:
        glPopMatrix()


@contextmanager
def emission(rgb: Sequence[float], color: Sequence[float] | None = None) -> Iterator[None]:
    """Sets GL_EMISSION (and optionally glColor) for the block, restoring both afterwards."""
    rgba = (*rgb[:3], 1.0) if len(rgb) < 4 else tuple(rgb)
    glMaterialfv(GL_FRONT, GL_EMISSION, rgba)
    if color is not None:
        glColor3f(*color[:3])
    try:
        yield
    finally:
        glMaterialfv(GL_FRONT, GL_EMISSION, NO_EMISSION)
        if color is not None:
            glColor3f(*WHITE)


@contextmanager
def specular(rgb: Sequence[float], shininess: float) -> Iterator[None]:
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*rgb[:3], 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
    try:
        yield
    finally:
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, NO_EMISSION)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)


@contextmanager
def unlit() -> Iterator[None]:
    glDisable(GL_LIGHTING)
    try:
        yield
    finally:
        glEnable(GL_LIGHTING)


def reset_emission() -> None:
    glMaterialfv(GL_FRONT, GL_EMISSION, NO_EMISSION)
