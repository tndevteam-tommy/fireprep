"""Minimal finite-state-machine scaffolding for NPC behaviour.

A ``State`` is a small object with ``enter`` / ``update`` / ``exit`` hooks. ``update`` returns
the next state to switch to (or ``None`` to stay). The machine tracks ``time_in_state`` so
states do not each need their own timer, and transitions always run ``exit`` then ``enter``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

A = TypeVar("A")  # the agent type the states operate on


class State(Generic[A]):
    name: str = "STATE"

    def enter(self, agent: A) -> None:  # noqa: B027 - intentionally optional hook
        pass

    def update(self, agent: A, dt: float) -> State[A] | None:
        return None

    def exit(self, agent: A) -> None:  # noqa: B027 - intentionally optional hook
        pass

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class StateMachine(Generic[A]):
    def __init__(self, agent: A, initial: State[A]) -> None:
        self.agent = agent
        self.state: State[A] = initial
        self.time_in_state = 0.0
        initial.enter(agent)

    @property
    def name(self) -> str:
        return self.state.name

    def transition_to(self, state: State[A]) -> None:
        self.state.exit(self.agent)
        self.state = state
        self.time_in_state = 0.0
        state.enter(self.agent)

    def reset(self, initial: State[A]) -> None:
        """Hard reset without running the outgoing state's ``exit`` hook."""
        self.state = initial
        self.time_in_state = 0.0
        initial.enter(self.agent)

    def update(self, dt: float) -> None:
        self.time_in_state += dt
        next_state = self.state.update(self.agent, dt)
        if next_state is not None:
            self.transition_to(next_state)
