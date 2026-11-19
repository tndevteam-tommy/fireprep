"""The game's full screens and overlays, drawn with ``render.ui``:

* the [Tab] screen - INVENTORY (a scrolling grid of item icons with their names under
  them, the selected item's details and an EQUIP / UNEQUIP button for cyber mods) and
  QUESTS (a scrolling list on the left; on the right the quest's objective, pictures of
  what it is about and its written details in a scrolling panel);
* the start menu, the pause menu and the load menu;
* captions (dialogue and remarks), notices, the [E] prompt;
* being caught: the red "you were caught" screen, the fade to black and back.

Layouts are worked out from the screen size every frame; the clickable rectangles are
left on the screen/menu objects (``hotspots``) for the next frame's clicks.
"""

from __future__ import annotations

import math

from cyberduke.config import SIMULATION
from cyberduke.core.inventory import KINDS
from cyberduke.core.screens import GRID_COLUMNS, TABS, Hotspot, TabScreen
from cyberduke.render.ui import (
    CYAN,
    GLASS_LIGHT,
    MAGENTA,
    RED,
    T_CYAN,
    T_DIM,
    T_FAINT,
    T_GREEN,
    T_MAGENTA,
    T_OFF,
    T_RED,
    T_WHITE,
    T_YELLOW,
    YELLOW,
    Ui,
)

HINT_TAIL = ("[ ]  SWITCH TAB", "[TAB]  CLOSE")  # the key hints under both tabs end alike
SLOT_LABEL_LINES = 2
SLOT_LABEL_SIZE = 11


def _backdrop(ui: Ui, w: float, h: float, alpha: float = 0.86) -> None:
    ctx = ui.ctx
    ctx.rect(0, 0, w, h, (0.0, 0.015, 0.03, alpha))
    ui.scanlines_over(0, 0, w, h, 0.1)
    for k in range(6):  # a faint vignette at the edges
        a = 0.05 * (6 - k)
        ctx.rect(0, k * 6, w, 6, (0.0, 0.0, 0.0, a))
        ctx.rect(0, h - (k + 1) * 6, w, 6, (0.0, 0.0, 0.0, a))


# ============================================================================ the tab screen
def draw_tab_screen(ui: Ui, w: float, h: float, sim, screen: TabScreen, icons, pictures) -> None:
    """``icons.get(name) -> texture``; ``pictures(picture) -> texture or None``."""
    _backdrop(ui, w, h)
    hot: list[Hotspot] = []
    # the header: the deck's name, the tabs, money and cyberware
    ui.text("// NETRUNNER DECK", 40, 26, 12, T_FAINT)
    tab_w, tab_h = 190.0, 34.0
    x0 = w / 2 - len(TABS) * (tab_w + 12) / 2
    for k, name in enumerate(TABS):
        x = x0 + k * (tab_w + 12)
        on = k == screen.tab
        ui.button(
            x, 18, tab_w, tab_h, name, hot=on, accent=CYAN if on else (0.0, 0.5, 0.6, 0.8), size=16
        )
        hot.append(Hotspot(x, 18, tab_w, tab_h, "tab", k))
    ui.text_right("[", x0 - 14, 26, 18, T_MAGENTA)
    ui.text("]", x0 + len(TABS) * (tab_w + 12) + 2, 26, 18, T_MAGENTA)
    inv = sim.player.inventory
    ui.text_right(f"EDDIES  {inv.count('eddies'):,}", w - 40, 20, 14, T_YELLOW)
    installed = len(inv.equipped)
    ui.text_right(f"CYBERWARE INSTALLED  {installed}", w - 40, 38, 11, T_MAGENTA)
    ui.divider(40, 64, w - 80)
    top, bottom = 84.0, h - 52.0
    if TABS[screen.tab] == "INVENTORY":
        _inventory(ui, w, top, bottom, sim, screen, icons, hot)
        hint = "     ".join(
            ("ARROWS / WASD  SELECT", "[ENTER] / [E]  EQUIP / UNEQUIP", "WHEEL  SCROLL", *HINT_TAIL)
        )
    else:
        _quests(ui, w, top, bottom, sim, screen, pictures, hot)
        hint = "     ".join(
            ("[UP] / [DOWN]  SELECT QUEST", "WHEEL / [PGUP] [PGDN]  SCROLL DETAILS", *HINT_TAIL)
        )
    ui.divider(40, h - 40, w - 80)
    ui.text_center(ui.fit(hint, w - 80, 11), w / 2, h - 32, 11, T_DIM)
    screen.hotspots = hot


def _inventory(ui: Ui, w, top, bottom, sim, screen: TabScreen, icons, hot) -> None:
    inv = sim.player.inventory
    grid_x, grid_w = 40.0, w * 0.56
    ui.panel(grid_x, top, grid_w, bottom - top, title=f"CARRIED  {len(inv)}/{inv.capacity}")
    pad, gap = 18.0, 12.0
    cell = (grid_w - 2 * pad - (GRID_COLUMNS - 1) * gap - 12) / GRID_COLUMNS
    label_h = SLOT_LABEL_LINES * (SLOT_LABEL_SIZE + 3) + 6
    row_h = cell + label_h + gap
    view_top, view_h = top + pad, bottom - top - 2 * pad
    screen.grid_rows_visible = max(1, int((view_h + gap) // row_h))
    rows = (len(inv) + GRID_COLUMNS - 1) // GRID_COLUMNS
    ctx = ui.ctx
    ctx.clip(grid_x + 2, view_top - 4, grid_w - 4, view_h + 8)
    for k, stack in enumerate(inv.stacks):
        row, col = divmod(k, GRID_COLUMNS)
        if row < screen.grid_scroll or row >= screen.grid_scroll + screen.grid_rows_visible + 1:
            continue
        x = grid_x + pad + col * (cell + gap)
        y = view_top + (row - screen.grid_scroll) * row_h
        selected = k == screen.slot
        _slot(ui, x, y, cell, stack, selected, icons)
        if y + cell <= view_top + view_h:
            hot.append(Hotspot(x, y, cell, cell + label_h, "slot", k))
    ctx.unclip()
    ui.scrollbar(
        grid_x + grid_w - 12, view_top, view_h, rows, screen.grid_rows_visible, screen.grid_scroll
    )
    if not inv.stacks:
        ui.text_center("NOTHING ON YOU", grid_x + grid_w / 2, (top + bottom) / 2, 16, T_FAINT)
    # the selected item
    dx = grid_x + grid_w + 24
    dw = w - 40 - dx
    ui.panel(dx, top, dw, bottom - top, accent=MAGENTA, title="DETAILS", title_color=T_MAGENTA)
    if not inv.stacks:
        return
    stack = inv.stacks[min(screen.slot, len(inv) - 1)]
    item = stack.item
    big = min(128.0, dw * 0.42)
    ctx.rect(dx + 20, top + 24, big, big, (0.0, 0.04, 0.06, 0.95))
    ctx.rect_outline(dx + 20, top + 24, big, big, MAGENTA if stack.equipped else CYAN, 1.0)
    ctx.image(icons.get(item.icon), dx + 20, top + 24, big, big)
    tx, tw = dx + 32 + big, dw - big - 52
    y = top + 26
    for line in ui.wrap(stack.name, tw, 17, max_lines=3):
        ui.text(line, tx, y, 17, T_WHITE)
        y += 21
    ui.text(KINDS.get(item.kind, item.kind.upper()), tx, y + 4, 12, T_MAGENTA)
    y += 22
    if item.slot:
        ui.text(f"SLOT: {item.slot}", tx, y, 11, T_DIM)
        y += 16
    if item.stackable:
        ui.text(f"QUANTITY: {stack.count:,}", tx, y, 11, T_DIM)
        y += 16
    if item.value:
        ui.text(f"WORTH: {item.value:,} EDDIES", tx, y, 11, T_DIM)
    y = max(y + 18, top + 40 + big)
    ui.divider(dx + 20, y, dw - 40)
    y += 12
    for line in item.effect_lines():
        ui.text(f"+ {line}" if "+" in line else f"- {line}", dx + 20, y, 14, T_GREEN)
        y += 20
    if item.effects:
        y += 6
    for line in ui.wrap(item.description, dw - 40, 13):
        if y > bottom - 90:
            break
        ui.text(line, dx + 20, y, 13, T_DIM)
        y += 17
    if item.equippable:
        status = "INSTALLED - ACTIVE" if stack.equipped else "NOT INSTALLED"
        ui.text(status, dx + 20, bottom - 92, 12, T_GREEN if stack.equipped else T_FAINT)
        bw, bh = dw - 40, 40.0
        by = bottom - 66
        label = "UNEQUIP" if stack.equipped else "EQUIP"
        ui.button(
            dx + 20,
            by,
            bw,
            bh,
            f"[ENTER]  {label}",
            hot=True,
            accent=MAGENTA if stack.equipped else CYAN,
            size=15,
        )
        hot.append(Hotspot(dx + 20, by, bw, bh, "equip", screen.slot))


def _slot(ui: Ui, x: float, y: float, cell: float, stack, selected: bool, icons) -> None:
    ctx = ui.ctx
    item = stack.item
    ctx.rect(x, y, cell, cell, GLASS_LIGHT)
    if selected:
        pulse = 0.6 + 0.4 * math.sin(ui.time * 5.0)
        ctx.rect(x, y, cell, cell, (0.0, 0.5 * pulse, 0.6 * pulse, 0.35))
    ctx.rect_outline(
        x, y, cell, cell, CYAN if selected else (0.0, 0.45, 0.55, 0.7), 2.0 if selected else 1.0
    )
    if selected:
        ui.brackets(x - 4, y - 4, cell + 8, cell + 8, YELLOW, arm=10)
    m = cell * 0.1
    ctx.image(icons.get(item.icon), x + m, y + m, cell - 2 * m, cell - 2 * m)
    if stack.equipped:
        ctx.rect(x + cell - 20, y + 4, 16, 16, MAGENTA)
        ui.text("E", x + cell - 16, y + 4, 12, T_WHITE)
    if item.stackable and stack.count > 1:
        ui.text_right(f"x{stack.count}", x + cell - 6, y + cell - 18, 12, T_YELLOW)
    ly = y + cell + 4
    for line in ui.wrap(stack.name, cell, SLOT_LABEL_SIZE, max_lines=SLOT_LABEL_LINES):
        ui.text_center(line, x + cell / 2, ly, SLOT_LABEL_SIZE, T_WHITE if selected else T_DIM)
        ly += SLOT_LABEL_SIZE + 3


def _quests(ui: Ui, w, top, bottom, sim, screen: TabScreen, pictures, hot) -> None:
    quests = sim.quests.quests
    lx, lw = 40.0, w * 0.3
    ui.panel(lx, top, lw, bottom - top, title=f"QUESTS  {len(quests)}")
    row_h = 64.0
    view_top, view_h = top + 18, bottom - top - 36
    screen.list_rows_visible = max(1, int(view_h // row_h))
    ctx = ui.ctx
    ctx.clip(lx + 2, view_top - 2, lw - 4, view_h + 4)
    for k, state in enumerate(quests):
        if k < screen.list_scroll or k >= screen.list_scroll + screen.list_rows_visible + 1:
            continue
        y = view_top + (k - screen.list_scroll) * row_h
        selected = k == screen.quest
        ctx.rect(
            lx + 14,
            y,
            lw - 40,
            row_h - 8,
            (0.0, 0.3, 0.4, 0.35) if selected else (0.0, 0.05, 0.08, 0.6),
        )
        ctx.rect(lx + 14, y, 3, row_h - 8, YELLOW if selected else CYAN)
        q = state.quest
        ui.text(ui.fit(q.title, lw - 70, 15), lx + 26, y + 6, 15, T_WHITE if selected else T_CYAN)
        tag, color = ("COMPLETE", T_GREEN) if state.done else ("ACTIVE", T_YELLOW)
        ui.text(ui.fit(f"{tag}  |  {q.giver}", lw - 70, 11), lx + 26, y + 28, 11, color)
        hot.append(Hotspot(lx + 14, y, lw - 40, row_h - 8, "quest", k))
    ctx.unclip()
    ui.scrollbar(
        lx + lw - 14, view_top, view_h, len(quests), screen.list_rows_visible, screen.list_scroll
    )
    rx = lx + lw + 24
    rw = w - 40 - rx
    if not quests:
        ui.panel(rx, top, rw, bottom - top, accent=MAGENTA)
        ui.text_center("NO QUESTS YET", rx + rw / 2, (top + bottom) / 2 - 20, 20, T_DIM)
        ui.text_center(
            "PEOPLE ON THE STREET MIGHT HAVE WORK FOR A NETRUNNER - TALK TO THEM [E]",
            rx + rw / 2,
            (top + bottom) / 2 + 10,
            11,
            T_FAINT,
        )
        return
    state = quests[min(screen.quest, len(quests) - 1)]
    q = state.quest
    ui.panel(rx, top, rw, bottom - top, accent=MAGENTA, title="QUEST", title_color=T_MAGENTA)
    y = top + 20
    ui.text(q.title, rx + 20, y, 24, T_WHITE)
    tag, color = ("COMPLETE", T_GREEN) if state.done else ("ACTIVE", T_YELLOW)
    ui.text_right(tag, rx + rw - 20, y + 6, 13, color)
    y += 32
    ui.text(ui.fit(f"GIVEN BY {q.giver}  |  {q.location}", rw - 40, 11), rx + 20, y, 11, T_DIM)
    y += 22
    # the current objective
    objective = sim.director.objective(state)
    obj_lines = ui.wrap(objective, rw - 70, 14, max_lines=2)
    box_h = 28 + 18 * len(obj_lines)
    ctx.rect(rx + 20, y, rw - 40, box_h, (0.25, 0.2, 0.0, 0.35))
    ctx.rect(rx + 20, y, 3, box_h, YELLOW)
    ui.text("CURRENT OBJECTIVE", rx + 32, y + 5, 10, T_YELLOW)
    for k, line in enumerate(obj_lines):
        ui.text(line, rx + 32, y + 20 + 18 * k, 14, T_WHITE)
    y += box_h + 14
    # pictures of what it is about
    pics = q.pictures
    if pics:
        n = len(pics)
        pw_draw = (rw - 40 - 16 * (n - 1)) / n
        ph = min(pw_draw * 0.5625, (bottom - y) * 0.42)
        pw_draw = ph / 0.5625
        for k, pic in enumerate(pics):
            px = rx + 20 + k * (pw_draw + 16)
            tex = pictures(pic)
            ctx.rect(px, y, pw_draw, ph, (0.0, 0.0, 0.0, 1.0))
            if tex is not None:
                ctx.image(tex, px, y, pw_draw, ph)
                ui.scanlines_over(px, y, pw_draw, ph, 0.12)
            else:
                ui.text_center("NO SIGNAL", px + pw_draw / 2, y + ph / 2 - 8, 14, T_FAINT)
            ctx.rect_outline(px, y, pw_draw, ph, CYAN, 1.0)
            ui.brackets(px, y, pw_draw, ph, CYAN, arm=12)
            rec = (1.0, 0.2, 0.2, 1.0) if int(ui.time * 2) % 2 == 0 else (0.4, 0.1, 0.1, 1.0)
            ctx.rect(px + 8, y + 8, 7, 7, rec)
            ui.text(ui.fit(pic.caption, pw_draw - 12, 11), px + 4, y + ph + 4, 11, T_CYAN)
        y += ph + 26
    # the written details: a scrolling panel
    view_top, view_h = y, bottom - 18 - y
    paras: list[tuple[str, int, tuple]] = []
    for para in q.description:
        paras.append((para, 13, T_WHITE))
    paras.append(("LOG", 11, T_MAGENTA))
    for t, text in state.journal:
        paras.append((f"[{int(t // 60):02d}:{int(t % 60):02d}]  {text}", 12, T_DIM))
    text_w = rw - 62
    lines: list[tuple[str, int, tuple, float]] = []
    for text, size, color in paras:
        for line in ui.wrap(text, text_w, size):
            lines.append((line, size, color, size + 5.0))
        lines.append(("", size, color, 8.0))
    total = sum(lh for *_, lh in lines)
    screen.detail_height, screen.detail_view = total, max(1.0, view_h)
    screen.scroll_detail(0.0)
    ctx.rect(rx + 20, view_top, rw - 40, view_h, (0.0, 0.02, 0.04, 0.6))
    ctx.clip(rx + 20, view_top, rw - 40, view_h)
    yy = view_top + 8 - screen.detail_scroll
    for line, size, color, lh in lines:
        if view_top - lh < yy < view_top + view_h:
            ui.text(line, rx + 30, yy, size, color)
        yy += lh
    ctx.unclip()
    ui.scrollbar(rx + rw - 30, view_top + 4, view_h - 8, total, view_h, screen.detail_scroll)
    hot.append(Hotspot(rx + 20, view_top, rw - 40, view_h, "detail"))


# ============================================================================ menus
def _message_color(message: str):
    """The game's one-line reports under a menu: failures red, anything else green."""
    return T_RED if message.startswith("COULD NOT") else T_GREEN


def draw_menu(ui: Ui, w: float, h: float, menu, subtitle: str = "", message: str = "") -> None:
    """The pause menu (and the start menu's panel): rows, values, the selected row lit,
    hotspots for the mouse."""
    ctx = ui.ctx
    items = menu.items
    panel_w, row_h = 460.0, 40.0
    panel_h = 96 + row_h * len(items)
    x, y = w / 2 - panel_w / 2, h / 2 - panel_h / 2
    if menu.title != "CYBERDUKE":
        ctx.rect(0, 0, w, h, (0.0, 0.01, 0.03, 0.72))
        ui.scanlines_over(0, 0, w, h, 0.08)
    ui.panel(x, y, panel_w, panel_h, fill=(0.01, 0.05, 0.08, 0.93))
    ui.glitch(menu.title, w / 2, y + 18, 24)
    if subtitle:
        ui.text_center(subtitle, w / 2, y + 50, 11, T_DIM)
    row_y = y + 74
    spots = []
    for i, item in enumerate(items):
        selected = i == menu.index
        if selected:
            ctx.rect(x + 16, row_y - 5, panel_w - 32, row_h - 6, (0.0, 0.55, 0.75, 0.3))
            ctx.rect(x + 16, row_y - 5, 3, row_h - 6, YELLOW)
        color = T_WHITE if selected else (T_DIM if item.enabled else T_OFF)
        ui.text(("> " if selected else "  ") + item.label, x + 30, row_y, 16, color)
        if item.value:
            ui.text_right(
                f"<  {item.value}  >",
                x + panel_w - 30,
                row_y + 1,
                14,
                T_YELLOW if selected else color,
            )
        if selected and item.note:
            ui.text_right(item.note.upper(), x + panel_w - 30, row_y + 3, 9, T_FAINT)
        spots.append((x + 16, row_y - 5, panel_w - 32, row_h - 6))
        row_y += row_h
    menu.hotspots = spots
    hint = "[UP]/[DOWN] SELECT   [LEFT]/[RIGHT] CHANGE   [ENTER] CONFIRM"
    if menu.title == "PAUSED":
        hint += "   [ESC] RESUME"
    ui.text_center(hint, w / 2, y + panel_h - 22, 10, T_FAINT)
    if message:
        ui.text_center(message, w / 2, y + panel_h + 16, 13, _message_color(message))


def draw_title(ui: Ui, w: float, h: float, menu, message: str = "") -> None:
    """The start menu: the street behind, the logo, the menu."""
    ctx = ui.ctx
    ctx.rect(0, 0, w, h, (0.0, 0.01, 0.03, 0.35))
    for k in range(30):  # darker towards the bottom
        ctx.rect(0, h * 0.55 + k * h * 0.015, w, h * 0.015 + 1, (0.0, 0.0, 0.02, 0.02 * k))
    ui.scanlines_over(0, 0, w, h, 0.07)
    cy = h * 0.16
    ui.glitch("CYBERDUKE", w / 2, cy, 64, (235, 250, 255))
    ui.text_center("T A C T I C A L   I N F I L T R A T I O N", w / 2, cy + 78, 14, T_MAGENTA)
    ctx.rect(w / 2 - 160, cy + 100, 320, 2, CYAN)
    # the menu rows, lower down
    items = menu.items
    row_h = 40.0
    panel_w = 420.0
    panel_h = 30 + row_h * len(items)
    x, y = w / 2 - panel_w / 2, h * 0.44
    ui.panel(x, y, panel_w, panel_h, fill=(0.01, 0.04, 0.07, 0.82))
    row_y = y + 18
    spots = []
    for i, item in enumerate(items):
        selected = i == menu.index
        if selected:
            ctx.rect(x + 16, row_y - 5, panel_w - 32, row_h - 6, (0.0, 0.55, 0.75, 0.3))
            ctx.rect(x + 16, row_y - 5, 3, row_h - 6, YELLOW)
        color = T_WHITE if selected else (T_DIM if item.enabled else T_OFF)
        ui.text(("> " if selected else "  ") + item.label, x + 30, row_y, 16, color)
        if item.value:
            ui.text_right(
                f"<  {item.value}  >",
                x + panel_w - 30,
                row_y + 1,
                14,
                T_YELLOW if selected else color,
            )
        spots.append((x + 16, row_y - 5, panel_w - 32, row_h - 6))
        row_y += row_h
    menu.hotspots = spots
    ui.text_center(
        "[UP]/[DOWN] SELECT   [LEFT]/[RIGHT] CHANGE   [ENTER] CONFIRM   OR USE THE MOUSE",
        w / 2,
        y + panel_h + 16,
        10,
        T_FAINT,
    )
    if message:
        ui.text_center(message, w / 2, y + panel_h + 40, 13, _message_color(message))
    ui.text("SAVES: ~/UNPREPARED/SAVES", 24, h - 28, 10, T_FAINT)


def draw_load_menu(ui: Ui, w: float, h: float, menu, message: str = "") -> None:
    ctx = ui.ctx
    ctx.rect(0, 0, w, h, (0.0, 0.01, 0.03, 0.75))
    ui.scanlines_over(0, 0, w, h, 0.08)
    panel_w = min(760.0, w - 80)
    row_h = 54.0
    panel_h = 120 + row_h * menu.VISIBLE
    x, y = w / 2 - panel_w / 2, h / 2 - panel_h / 2
    ui.panel(
        x,
        y,
        panel_w,
        panel_h,
        fill=(0.01, 0.05, 0.08, 0.94),
        title=f"SAVE FILES  {len(menu.saves)}",
    )
    ui.glitch("LOAD GAME", w / 2, y + 18, 24)
    top = y + 64
    spots = []
    ctx.clip(x + 2, top - 4, panel_w - 4, row_h * menu.VISIBLE + 4)
    for k in range(menu.scroll, min(len(menu.saves), menu.scroll + menu.VISIBLE)):
        info = menu.saves[k]
        ry = top + (k - menu.scroll) * row_h
        selected = k == menu.index
        ctx.rect(
            x + 16,
            ry,
            panel_w - 44,
            row_h - 8,
            (0.0, 0.35, 0.5, 0.4) if selected else (0.0, 0.05, 0.08, 0.6),
        )
        ctx.rect(x + 16, ry, 3, row_h - 8, YELLOW if selected else CYAN)
        ui.text(info.title, x + 30, ry + 6, 15, T_WHITE if selected else T_CYAN)
        ui.text_right(info.path.name, x + panel_w - 40, ry + 8, 10, T_FAINT)
        ui.text(ui.fit(info.detail, panel_w - 80, 11), x + 30, ry + 26, 11, T_DIM)
        spots.append((x + 16, ry, panel_w - 44, row_h - 8))
    ctx.unclip()
    ui.scrollbar(
        x + panel_w - 18, top, row_h * menu.VISIBLE - 8, len(menu.saves), menu.VISIBLE, menu.scroll
    )
    if not menu.saves:
        ui.text_center("NO SAVED GAMES YET", w / 2, top + 60, 16, T_FAINT)
    menu.hotspots = spots
    ui.text_center(
        "[UP]/[DOWN] SELECT   [ENTER] LOAD   WHEEL SCROLL   [ESC] BACK",
        w / 2,
        y + panel_h - 26,
        10,
        T_FAINT,
    )
    if message:
        ui.text_center(message, w / 2, y + panel_h + 14, 13, _message_color(message))


# ============================================================================ in game
def draw_captions(ui: Ui, w: float, h: float, sim) -> None:
    """Dialogue (and remarks shouted in passing) as captions along the bottom."""
    lines = []
    conv = sim.conversation
    if conv is not None:
        lines.append((conv.line.speaker, conv.shown_text, conv.typed, True))
    for bark in sim.barks:
        lines.append((bark.line.speaker, bark.shown_text, True, False))
    y = h - 130.0
    for speaker, text, typed, talking in reversed(lines):
        box_w = min(760.0, w - 120)
        wrapped = ui.wrap(text or " ", box_w - 40, 16, max_lines=3)
        box_h = 38 + 21 * len(wrapped)
        x = w / 2 - box_w / 2
        by = y - box_h
        ui.ctx.rect(x, by, box_w, box_h, (0.0, 0.02, 0.04, 0.78))
        ui.ctx.rect(x, by, 3, box_h, MAGENTA)
        ui.brackets(x, by, box_w, box_h, (0.0, 0.8, 0.9, 0.8), arm=10)
        tag_w = ui.width(speaker, 12) + 18
        ui.ctx.rect(x + 14, by - 9, tag_w, 18, MAGENTA)
        ui.text(speaker, x + 23, by - 8, 12, T_WHITE)
        for k, line in enumerate(wrapped):
            ui.text(line, x + 20, by + 16 + 21 * k, 16, T_WHITE)
        if talking and typed:
            blink = int(ui.time * 2.5) % 2 == 0
            ui.text_right(
                "[E] >" if blink else "[E]  ", x + box_w - 16, by + box_h - 20, 11, T_CYAN
            )
        y = by - 16


TOASTS_Y = 206.0  # below the level title card's band (hud.TITLE_Y + 46), with a gap


def draw_toasts(ui: Ui, w: float, h: float, sim) -> None:
    y = TOASTS_Y
    for text, left in sim.toasts:
        alpha = min(1.0, left / 0.6)
        tw = ui.width(text, 14) + 40
        x = w / 2 - tw / 2
        ui.ctx.rect(x, y, tw, 28, (0.0, 0.05, 0.08, 0.85 * alpha))
        ui.ctx.rect(x, y, tw, 2, (0.0, 0.95, 1.0, alpha))
        ui.ctx.rect(x, y + 26, tw, 2, (1.0, 0.2, 0.75, alpha))
        ui.text_center(text, w / 2, y + 6, 14, T_CYAN)
        y += 36


def draw_prompt(ui: Ui, w: float, h: float, text: str) -> None:
    if not text:
        return
    tw = ui.width(text, 13) + 24
    x, y = w / 2 - tw / 2, h / 2 + 58
    ui.ctx.rect(x, y, tw, 24, (0.0, 0.04, 0.06, 0.75))
    ui.ctx.rect_outline(x, y, tw, 24, CYAN, 1.0)
    ui.text_center(text, w / 2, y + 4, 13, T_WHITE)


def draw_caught(ui: Ui, w: float, h: float, sim) -> None:
    """Caught: the red screen (pulsing, "YOU WERE CAUGHT" and what happens now), then a
    fade to black; afterwards the picture fades back in from black. The same black fades
    the picture out at a level's last door and in again on the next level."""
    ctx = ui.ctx
    if sim.state.value == "CAUGHT":
        t = sim.state_timer
        hold = sim.caught_hold
        pulse = 0.5 + 0.5 * math.sin(t * 7.0)
        ctx.rect(0, 0, w, h, (0.55 + 0.15 * pulse, 0.02, 0.03, 0.55))
        for yy in range(0, int(h), 4):
            ctx.rect(0, yy, w, 1, (0.0, 0.0, 0.0, 0.18))
        band_y = h / 2 - 70
        ctx.rect(0, band_y, w, 140, (0.08, 0.0, 0.0, 0.7))
        ctx.rect(0, band_y, w, 3, RED)
        ctx.rect(0, band_y + 137, w, 3, RED)
        shake = 3.0 * math.sin(t * 41.0) if t < 0.5 else 0.0
        ui.text_center("YOU WERE CAUGHT", w / 2 + shake, band_y + 22, 44, (255, 70, 80))
        if sim.caught_rule == "evict":
            sub = "YOU'RE BEING THROWN OUT"
            if sim.caught_area is not None:
                sub = f"YOU'RE BEING THROWN OUT OF {sim.caught_area.name}"
        else:
            sub = "THE RUN IS OVER - A NEW GAME BEGINS"
        ui.text_center(sub, w / 2, band_y + 80, 16, (255, 200, 200))
        if sim.caught_by:
            ui.text_center(
                ui.fit(f"SPOTTED BY: {sim.caught_by}", w - 80, 12),
                w / 2,
                band_y + 106,
                12,
                (255, 150, 150),
            )
        if t > hold:
            fade = min(1.0, (t - hold) / SIMULATION.caught_fade)
            ctx.rect(0, 0, w, h, (0.0, 0.0, 0.0, fade))
    else:
        # out of a level's last door (fading out, then in on the next level), or back from
        # being caught (fading in)
        black = getattr(sim, "screen_black", 0.0)
        if black > 0.0:
            ctx.rect(0, 0, w, h, (0.0, 0.0, 0.0, black))
