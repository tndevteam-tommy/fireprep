"""Enforces the layering rule: simulation code never imports OpenGL or the render package."""

import ast
import pathlib

import cyberduke

SRC = pathlib.Path(cyberduke.__file__).parent
SIMULATION_PACKAGES = (
    "core", "world", "entities", "ai", "animation", "assets", "systems", "levels",
    "config.py", "simulation.py", "game.py",
)  # fmt: skip
FORBIDDEN_PREFIXES = ("OpenGL", "cyberduke.render", "pygame")


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _simulation_files():
    for entry in SIMULATION_PACKAGES:
        p = SRC / entry
        if p.is_file():
            yield p
        else:
            yield from p.rglob("*.py")


def test_simulation_packages_do_not_import_rendering_or_pygame():
    offenders = []
    for path in _simulation_files():
        for mod in _imports(path):
            if mod.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(SRC)} imports {mod}")
    assert not offenders, "\n".join(offenders)


def test_render_package_is_the_only_opengl_user():
    users = set()
    for path in SRC.rglob("*.py"):
        for mod in _imports(path):
            if mod.startswith("OpenGL"):
                users.add(path.relative_to(SRC).parts[0])
    assert users <= {"render"}, f"OpenGL imported outside render/: {sorted(users)}"
