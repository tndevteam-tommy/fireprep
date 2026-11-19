"""Things people handle (``render.items``): what they are drawn from. The fish is a fish
(a tapering body with a head, a forked tail and its fins, an eye), and the sushi cut from
it on the board is the sushi packed into the box - the same pieces, as many of them."""

from __future__ import annotations

import numpy as np

from cyberduke.animation.street import SUSHI_PIECES
from cyberduke.render import shapes
from cyberduke.render.context import DrawContext
from cyberduke.render.items import ITEMS, draw_item


class Recorder(DrawContext):
    """Records the shapes and boxes an item is drawn from (and their textures)."""

    name = "recorder"

    def __init__(self):
        self.drawn: list[tuple[str, object]] = []

    def shape(self, name, tex_id=None):
        self.drawn.append((name, tex_id))

    def cube(self, tex_id=None):
        self.drawn.append(("cube", tex_id))

    def cylinder(self, r_bot, r_top, h, slices=10, tex_id=None):
        self.drawn.append(("cylinder", tex_id))

    def begin_frame(self, eye, target): ...
    def end_frame(self): ...
    def push(self): ...
    def pop(self): ...
    def translate(self, x, y, z): ...
    def rotate(self, a, x, y, z): ...
    def scale(self, x, y, z): ...
    def color(self, r, g, b): ...
    def emission(self, rgb): ...
    def specular(self, rgb, shininess=0.0): ...
    def lines(self, segments, rgba, width=1.0): ...
    def soft_disc(self, rgba, slices=24, inner=0.0): ...
    def cached(self, key, build):
        build()

    def invalidate(self, key=None): ...
    def begin_overlay(self, width, height): ...
    def end_overlay(self): ...
    def rect(self, x, y, w, h, rgba): ...
    def rect_outline(self, x, y, w, h, rgba, width=2.0): ...
    def image(self, tex_id, x, y, w, h): ...
    def upload_texture_rgba(self, data, w, h):
        return 1

    def delete_texture(self, tex_id): ...


TEXTURES = {"fish_skin": 11, "fin": 12, "salmon": 13, "cardboard": 14, "metal": 15}


def _drawn(name: str) -> list[tuple[str, object]]:
    ctx = Recorder()
    draw_item(ctx, name, TEXTURES)
    return ctx.drawn


def test_the_fish_is_a_fish_not_a_few_boxes():
    drawn = _drawn("fish")
    assert ("fish_body", TEXTURES["fish_skin"]) in drawn, "a shaped, scaly body"
    fins = [d for d in drawn if d[0] == "fin"]
    assert len(fins) >= 8, "two tail lobes, dorsal, adipose, anal, pectoral and pelvic fins"
    assert all(tex == TEXTURES["fin"] for _, tex in fins)
    assert sum(1 for d in drawn if d[0] == "sphere") == 2, "an eye on each side"
    assert not any(d[0] == "cube" for d in drawn), "no boxes left in it"


def test_the_fish_body_tapers_to_the_tail_and_is_deepest_ahead_of_the_middle():
    v, _ = shapes.mesh("fish_body")
    pos = v[:, :3]
    assert pos.min() >= -0.5 - 1e-9 and pos.max() <= 0.5 + 1e-9

    rings = sorted({round(float(y), 4) for y in pos[:, 1]})

    def depth_at(y: float) -> float:  # (of the ring nearest y)
        near = min(rings, key=lambda r: abs(r - y))
        return float(np.ptp(pos[np.abs(pos[:, 1] - near) < 1e-4][:, 2]))

    tail, middle, ahead, head = depth_at(-0.48), depth_at(-0.06), depth_at(0.18), depth_at(0.44)
    assert tail < 0.4 * ahead, "a slim root to the tail"
    assert ahead >= middle and ahead > head, "deepest a little ahead of the middle"


def test_the_sushi_cut_on_the_board_is_what_goes_into_the_box():
    """The chef cuts the fish into the pieces he packs: the board shows as many nigiri
    as the box is filled with (and as the customer eats)."""

    def pieces(name: str) -> int:
        return sum(1 for d in _drawn(name) if d == ("plate", TEXTURES["salmon"]))

    assert pieces("sushi") == pieces("sushi_open") == SUSHI_PIECES
    for left in range(1, SUSHI_PIECES):
        assert pieces(f"sushi_open{left}") == left
    assert "sushi" in ITEMS and "fish" in ITEMS
