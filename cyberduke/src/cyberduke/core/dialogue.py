"""Captioned dialogue: who says what, one line at a time.

A ``Conversation`` is a list of ``Line``s (speaker, text). The HUD shows the current line
as a caption along the bottom of the screen, typing it out (``shown_text``); [E] skips to
the end of the line and then to the next one, and a line moves on by itself once it has
been up long enough to read. When the last line is done ``on_done`` runs (a quest starts,
an item changes hands).

``Barks`` are one-line remarks that need no answer ("Hey! Get out of my place!") - shown
the same way, never blocking anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

TYPE_RATE = 45.0  # characters per second the caption types out at
READ_TIME = 1.6  # seconds a fully typed line stays up before moving on by itself
PER_CHAR = 0.045  # ... plus this long per character


@dataclass(frozen=True)
class Line:
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        """How long the line stays up on its own: typed out, then time to read it."""
        return len(self.text) / TYPE_RATE + READ_TIME + PER_CHAR * len(self.text)


class Conversation:
    def __init__(
        self,
        lines: Sequence[Line],
        on_done: Callable[[], None] | None = None,
        partner: object | None = None,
    ) -> None:
        if not lines:
            raise ValueError("a conversation needs at least one line")
        self.lines = list(lines)
        self.on_done = on_done
        self.partner = partner  # who the player is talking to (they face the player)
        self.index = 0
        self.time = 0.0  # into the current line
        self.finished = False

    @property
    def line(self) -> Line:
        return self.lines[min(self.index, len(self.lines) - 1)]

    @property
    def typed(self) -> bool:
        return self.time * TYPE_RATE >= len(self.line.text)

    @property
    def shown_text(self) -> str:
        """The part of the current line typed out so far."""
        return self.line.text[: int(self.time * TYPE_RATE)]

    def advance(self) -> None:
        """[E]: finish typing the line, or go on to the next one."""
        if self.finished:
            return
        if not self.typed:
            self.time = len(self.line.text) / TYPE_RATE
            return
        self._next()

    def update(self, dt: float) -> None:
        if self.finished:
            return
        self.time += dt
        if self.time >= self.line.duration:
            self._next()

    def _next(self) -> None:
        self.index += 1
        self.time = 0.0
        if self.index >= len(self.lines):
            self.index = len(self.lines) - 1
            self.finished = True
            if self.on_done is not None:
                self.on_done()


class Bark:
    """A remark shouted in passing (no answer expected)."""

    def __init__(self, line: Line, seconds: float | None = None) -> None:
        self.line = line
        self.time = 0.0
        self.seconds = line.duration if seconds is None else seconds

    @property
    def shown_text(self) -> str:
        return self.line.text[: int(self.time * TYPE_RATE)]

    @property
    def done(self) -> bool:
        return self.time >= self.seconds

    def update(self, dt: float) -> None:
        self.time += dt
