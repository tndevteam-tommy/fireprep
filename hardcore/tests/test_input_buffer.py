"""Input buffer: fresh-press vs held semantics that make chords safe."""
from harness import *  # noqa


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _buf():
    clk = Clock()
    return InputBuffer(buffer_time=0.15, time_fn=clk), clk


def test_fresh_press_expires_after_buffer_window():
    buf, clk = _buf()
    buf.set_key_down(Action.JUMP)
    buf.set_key_up(Action.JUMP)
    clk.t = 0.10
    buf.update()
    assert buf.has_action(Action.JUMP, consume=False)
    clk.t = 0.20
    buf.update()
    assert not buf.has_action(Action.JUMP, consume=False)


def test_has_action_consumes_press():
    buf, clk = _buf()
    buf.set_key_down(Action.JUMP)
    assert buf.has_action(Action.JUMP)
    assert not buf.has_action(Action.JUMP), "a single press must only trigger once"


def test_modifier_accepts_recent_tap():
    buf, clk = _buf()
    buf.set_key_down(Action.GRAB)
    buf.set_key_up(Action.GRAB)          # tapped E
    clk.t = 0.05
    buf.update()
    assert buf.held_or_buffered(Action.GRAB)
    clk.t = 0.30
    buf.update()
    assert not buf.held_or_buffered(Action.GRAB)


def test_consume_clears_every_buffered_press_of_an_action():
    buf, clk = _buf()
    buf.set_key_down(Action.JUMP)
    buf.set_key_up(Action.JUMP)
    buf.set_key_down(Action.JUMP)
    buf.consume(Action.JUMP)
    assert not buf.has_action(Action.JUMP, consume=False)
    assert buf.held_or_buffered(Action.JUMP), "consuming the press does not release the key"
