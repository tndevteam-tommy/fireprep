from unprepared.core.messages import MessageLog


def test_log_is_bounded():
    log = MessageLog(max_messages=3)
    for i in range(5):
        log.add(f"m{i}")
    assert log.messages == ["m2", "m3", "m4"]
    assert len(log) == 3


def test_last_and_iteration():
    log = MessageLog(6)
    for i in range(6):
        log(f"m{i}")          # callable alias
    assert log.last(2) == ["m4", "m5"]
    assert log.last(0) == []
    assert list(log) == [f"m{i}" for i in range(6)]


def test_echo_callback_receives_every_message():
    seen = []
    log = MessageLog(2, echo=seen.append)
    log.add("a"); log.add("b"); log.add("c")
    assert seen == ["a", "b", "c"]
    assert log.messages == ["b", "c"]


def test_messages_returns_copy():
    log = MessageLog(3)
    log.add("x")
    log.messages.append("y")
    assert log.messages == ["x"]
