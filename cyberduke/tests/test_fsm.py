from cyberduke.ai.fsm import State, StateMachine


class Log:
    def __init__(self):
        self.events = []


class A(State[Log]):
    name = "A"

    def enter(self, agent):
        agent.events.append("enter A")

    def update(self, agent, dt):
        agent.events.append(f"update A {dt}")
        return B() if dt > 0.5 else None

    def exit(self, agent):
        agent.events.append("exit A")


class B(State[Log]):
    name = "B"

    def enter(self, agent):
        agent.events.append("enter B")


def test_initial_state_is_entered_and_time_accumulates():
    log = Log()
    fsm = StateMachine(log, A())
    assert fsm.name == "A" and log.events == ["enter A"]
    fsm.update(0.1)
    fsm.update(0.1)
    assert fsm.time_in_state == 0.2


def test_transition_runs_exit_then_enter_and_resets_timer():
    log = Log()
    fsm = StateMachine(log, A())
    fsm.update(0.1)
    fsm.update(1.0)  # A asks for B
    assert fsm.name == "B"
    assert log.events == ["enter A", "update A 0.1", "update A 1.0", "exit A", "enter B"]
    assert fsm.time_in_state == 0.0


def test_reset_skips_exit_hook():
    log = Log()
    fsm = StateMachine(log, A())
    fsm.update(0.3)
    fsm.reset(B())
    assert fsm.name == "B" and "exit A" not in log.events
    assert fsm.time_in_state == 0.0


def test_repr():
    assert repr(A()) == "<A A>"
