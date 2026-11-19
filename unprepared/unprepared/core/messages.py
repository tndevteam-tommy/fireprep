"""Bounded in-game message/dialogue log (replaces the old module-level global)."""
from __future__ import annotations

from typing import Callable, List, Optional


class MessageLog:
    """The last ``max_messages`` lines. Each line has a *kind*: ``"event"`` (what concerns the
    player - the default) or ``"news"`` (the town's comings and goings, the hour striking), and a
    running serial number, so a reader can tell what is new since it last looked (``since``)."""

    NEWS = "news"
    EVENT = "event"

    def __init__(self, max_messages: int = 6, echo: Optional[Callable[[str], None]] = None):
        self.max_messages = max_messages
        self._messages: List[str] = []
        self._kinds: List[str] = []
        self._echo = echo
        self.serial = 0                  # how many lines have ever been added

    def add(self, msg: str, kind: str = EVENT) -> None:
        self._messages.append(msg)
        self._kinds.append(kind)
        self.serial += 1
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)
            self._kinds.pop(0)
        if self._echo:
            self._echo(msg)

    def news(self, msg: str) -> None:
        """Add a line of the town's news (a notify callback in its own right)."""
        self.add(msg, self.NEWS)

    def since(self, serial: int) -> List[tuple]:
        """(text, kind) of the lines added after the log's ``serial`` was ``serial`` (those still kept)."""
        count = min(len(self._messages), max(0, self.serial - serial))
        if count == 0:
            return []
        return list(zip(self._messages[-count:], self._kinds[-count:]))

    # Allow the log itself to be used as a ``notify_callback``.
    __call__ = add

    @property
    def messages(self) -> List[str]:
        return list(self._messages)

    def last(self, count: int) -> List[str]:
        return self._messages[-count:] if count > 0 else []

    def clear(self) -> None:
        self._messages.clear()
        self._kinds.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)


def stdout_echo(msg: str) -> None:
    print(f"[HUD / DIALOGUE] {msg}")
