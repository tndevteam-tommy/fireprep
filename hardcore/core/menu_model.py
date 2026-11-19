"""Pause-menu selection model (no rendering, no pygame): drives the GL PauseMenu."""


class MenuModel:
    RESUME = "resume"
    CONTROLLER = "controller"
    FULLSCREEN = "fullscreen"
    MULTIPLAYER = "multiplayer"
    QUIT = "quit"

    def __init__(self, items=(("Resume", RESUME), ("Quit", QUIT))):
        self.items = list(items)
        self.selected = 0
        self.visible = False
        self.rects = {}        # item index -> (x, y, w, h) in screen pixels, filled by the renderer

    def open(self):
        self.visible = True
        self.selected = 0

    def close(self):
        self.visible = False

    def hit(self, pos, pad=(20, 8)):
        for i, (x, y, w, h) in self.rects.items():
            if x - pad[0] <= pos[0] <= x + w + pad[0] and y - pad[1] <= pos[1] <= y + h + pad[1]:
                return i
        return None

    def handle(self, action, pos=None):
        """
        action: "up" | "down" | "confirm" | "back" | "hover" | "click".
        Returns the activated item id (RESUME / QUIT) or None.
        """
        if not self.visible:
            return None
        if action == "back":
            return self.RESUME
        if action == "up":
            self.selected = (self.selected - 1) % len(self.items)
        elif action == "down":
            self.selected = (self.selected + 1) % len(self.items)
        elif action == "confirm":
            return self.items[self.selected][1]
        elif action == "hover":
            i = self.hit(pos)
            if i is not None:
                self.selected = i
        elif action == "click":
            i = self.hit(pos)
            if i is not None:
                self.selected = i
                return self.items[i][1]
        return None
