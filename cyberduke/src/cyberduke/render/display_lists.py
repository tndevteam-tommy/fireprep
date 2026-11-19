"""Display-list helpers: compile a draw callable once, replay it with one GL call.

Fixed-function display lists replay *exactly* the GL calls that were made while compiling,
so any immediate-mode draw function can be cached as long as what it draws is static.
Nested compilation is not allowed by GL, hence the guard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from OpenGL.GL import GL_COMPILE, glCallList, glDeleteLists, glEndList, glGenLists, glNewList

_compiling = False


def is_compiling() -> bool:
    return _compiling


def compile_list(draw: Callable[[], None]) -> int:
    """Runs ``draw`` inside glNewList/glEndList and returns the list id."""
    global _compiling
    if _compiling:
        raise RuntimeError("display lists cannot be compiled while another list is compiling")
    list_id = int(glGenLists(1))
    _compiling = True
    try:
        glNewList(list_id, GL_COMPILE)
        draw()
        glEndList()
    finally:
        _compiling = False
    return list_id


class ListCache:
    """Lazily compiles and caches one display list per key; ``invalidate`` to rebuild."""

    def __init__(self) -> None:
        self._lists: dict[Any, int] = {}

    def draw(self, key: Any, build: Callable[[], None]) -> None:
        list_id = self._lists.get(key)
        if list_id is None:
            list_id = compile_list(build)
            self._lists[key] = list_id
        glCallList(list_id)

    def invalidate(self, key: Any = None) -> None:
        keys = [key] if key is not None else list(self._lists)
        for k in keys:
            list_id = self._lists.pop(k, None)
            if list_id is not None:
                glDeleteLists(list_id, 1)

    def __len__(self) -> int:
        return len(self._lists)
