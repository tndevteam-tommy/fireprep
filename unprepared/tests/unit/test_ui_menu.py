"""Menus are pure data: navigation wraps, activation respects ``enabled``, long lists scroll."""
from unprepared.ui import Menu, MenuItem, slider_text


def test_scrolling_window_follows_the_cursor():
    menu = Menu("SAVES", [MenuItem(f"save {i}") for i in range(25)], visible_rows=10)
    assert menu.visible() == (0, menu.items[:10])
    for _ in range(12):
        menu.move(+1)
    first, rows = menu.visible()
    assert first == 3 and rows == menu.items[3:13] and menu.cursor == 12
    menu.move(-1)                                    # cursor wraps to... no: 11, still inside the window
    assert menu.visible()[0] == 3
    menu.cursor = 0
    assert menu.visible()[0] == 0
    menu.move(-1)                                    # wraps to the last item
    assert menu.cursor == 24 and menu.visible()[0] == 15


def test_short_menus_do_not_scroll():
    menu = Menu("M", [MenuItem("a"), MenuItem("b")])
    assert menu.visible() == (0, menu.items)


def test_activation_and_adjust():
    hits = []
    menu = Menu("M", [MenuItem("go", lambda: hits.append("go")),
                      MenuItem("off", lambda: hits.append("off"), enabled=lambda: False),
                      MenuItem("slider", adjust=lambda d: hits.append(d), value=lambda: "v")])
    menu.activate()
    menu.move(+1); menu.activate()                   # disabled: nothing
    menu.move(+1); menu.activate(); menu.adjust(-1)  # adjust items activate with +1
    assert hits == ["go", 1, -1]
    assert menu.current.display_value() == "v"
    assert slider_text(60, 50, 100, width=10, suffix=" deg") == "[##........] 60 deg"
