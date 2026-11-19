"""Display-list caching logic, with the GL entry points stubbed (no context needed)."""

import pytest

from cyberduke.render import display_lists as dl
from cyberduke.render import primitives


class FakeGL:
    def __init__(self):
        self.next_id = 1
        self.calls = []

    def glGenLists(self, n):
        self.next_id += 1
        return self.next_id - 1

    def glNewList(self, list_id, mode):
        self.calls.append(("new", list_id))

    def glEndList(self):
        self.calls.append(("end",))

    def glCallList(self, list_id):
        self.calls.append(("call", list_id))

    def glDeleteLists(self, list_id, n):
        self.calls.append(("delete", list_id))


@pytest.fixture
def gl(monkeypatch):
    fake = FakeGL()
    for name in ("glGenLists", "glNewList", "glEndList", "glCallList", "glDeleteLists"):
        monkeypatch.setattr(dl, name, getattr(fake, name))
    return fake


def test_compile_list_wraps_the_draw_callable(gl):
    drawn = []
    list_id = dl.compile_list(lambda: drawn.append("geometry"))
    assert list_id == 1 and drawn == ["geometry"]
    assert gl.calls == [("new", 1), ("end",)]
    assert not dl.is_compiling()


def test_nested_compilation_is_rejected_and_state_is_restored(gl):
    def inner():
        dl.compile_list(lambda: None)

    with pytest.raises(RuntimeError, match="cannot be compiled"):
        dl.compile_list(inner)
    assert not dl.is_compiling()


def test_list_cache_builds_once_then_replays(gl):
    cache = dl.ListCache()
    builds = []
    for _ in range(3):
        cache.draw("sector", lambda: builds.append(1))
    assert len(builds) == 1 and len(cache) == 1
    assert gl.calls.count(("call", 1)) == 3


def test_list_cache_invalidate_deletes_and_rebuilds(gl):
    cache = dl.ListCache()
    cache.draw("a", lambda: None)
    cache.draw("b", lambda: None)
    cache.invalidate("a")
    assert ("delete", 1) in gl.calls and len(cache) == 1
    cache.invalidate()
    assert len(cache) == 0
    cache.draw("a", lambda: None)  # rebuilt with a fresh id
    assert gl.calls[-1] == ("call", 3)


def test_primitives_fall_back_to_immediate_mode_when_not_compiled(monkeypatch):
    emitted = []
    monkeypatch.setattr(primitives, "glBegin", lambda mode: emitted.append("begin"))
    monkeypatch.setattr(primitives, "glEnd", lambda: emitted.append("end"))
    for name in (
        "glEnable",
        "glDisable",
        "glBindTexture",
        "glNormal3f",
        "glTexCoord2f",
        "glVertex3f",
    ):
        monkeypatch.setattr(primitives, name, lambda *a: None)
    monkeypatch.setattr(primitives, "glCallList", lambda i: emitted.append(("call", i)))
    primitives.reset_primitives()
    primitives.draw_cube(None)
    assert emitted == ["begin", "end"], "uncompiled: one immediate-mode quad batch"
    emitted.clear()
    primitives._cube_list = 7  # pretend compile_primitives() ran
    try:
        primitives.draw_cube(None)
        assert emitted == [("call", 7)], "compiled: one glCallList"
    finally:
        primitives.reset_primitives()


def test_closing_the_window_forgets_the_primitive_lists(monkeypatch):
    """Switching windowed <-> fullscreen closes the window and its GL context: list ids
    compiled in the old context must not be replayed in the new one (there they name the
    new context's own lists - a sector list calling itself never finishes drawing)."""
    from cyberduke.render import primitives, scene

    calls = []
    monkeypatch.setattr(scene.pygame.display, "quit", lambda: calls.append("quit"))
    monkeypatch.setattr(scene.pygame.display, "init", lambda: calls.append("init"))
    monkeypatch.setattr(primitives, "_cube_list", 7)
    scene.close_window()
    assert calls == ["quit", "init"]
    assert primitives._cube_list is None and not primitives._cylinder_lists
