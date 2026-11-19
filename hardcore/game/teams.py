"""Teams: red and blue.  Humans pick theirs; bots fill in to keep the sides even."""

RED = "red"
BLUE = "blue"
TEAMS = (RED, BLUE)
NAMES = {RED: "Red", BLUE: "Blue"}
SHIRTS = {RED: (0.86, 0.2, 0.18), BLUE: (0.18, 0.42, 0.95)}       # avatar shirts (0..1)
UI = {RED: (255, 112, 100), BLUE: (112, 164, 255)}                 # HUD text (0..255)


def valid(team):
    return team in TEAMS


def other(team):
    return BLUE if team == RED else RED


def name(team):
    return NAMES.get(team, "")


def smaller_team(counts, prefer=None):
    """The team with fewer players ({team: n}); `prefer` breaks a tie."""
    r, b = counts.get(RED, 0), counts.get(BLUE, 0)
    if r < b:
        return RED
    if b < r:
        return BLUE
    return prefer if valid(prefer) else RED
