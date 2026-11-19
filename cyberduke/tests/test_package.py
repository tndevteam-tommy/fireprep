"""Package-level sanity checks."""

import importlib

import cyberduke


def test_version_string():
    assert isinstance(cyberduke.__version__, str)


def test_logic_modules_import_without_a_gl_context():
    # Importing must never require a live OpenGL context (that would make
    # headless testing impossible). Modules that merely *import* OpenGL.GL are fine.
    for name in (
        "animation",
        "ai",
        "world",
        "entities",
        "assets.textures",
        "config",
        "systems",
        "simulation",
    ):
        importlib.import_module(f"cyberduke.{name}")
