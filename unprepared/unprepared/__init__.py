"""Unprepared - a rogue-like stealth & theft RPG.

Package layout
--------------
core/       pure helpers: 3D math, message log, day/night cycle
entities/   game objects: Item, Door, Player, NPC, SkillTree
world/      level data, World container, save/load
systems/    gameplay rules operating on entities (movement, trading, ...)
game.py     the headless simulation: state + action API (no pygame/OpenGL)
input/      pygame event -> Game action translation
rendering/  OpenGL presentation of a Game
app.py      window creation and the main loop
"""
__version__ = "0.2.0"


def _quiet_opengl() -> None:
    """PyOpenGL checks for a GL error after every call it makes - a round trip to the driver each
    time, and the renderer makes tens of thousands of calls a frame: a fifth of the frame went on
    asking. It must be switched off before ``OpenGL.GL`` is first imported (which the rendering
    modules do); set ``UNPREPARED_GL_DEBUG=1`` to keep the checks while hunting a GL bug."""
    import os
    if os.environ.get("UNPREPARED_GL_DEBUG"):
        return
    try:
        import OpenGL
    except ImportError:                      # (headless without PyOpenGL: nothing to quieten)
        return
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False


_quiet_opengl()
