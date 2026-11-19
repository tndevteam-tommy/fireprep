"""Procedural texture tooling (tools/textures) and the game's texture loader."""
import os

import pygame

from tools.textures.generators import SIZE, TEXTURE_KEYS, generate_texture, write_textures
from unprepared.rendering.textures import DEFAULT_TEXTURE_KEYS, placeholder_surface


def _bytes(surface):
    return pygame.image.tostring(surface, "RGB")


def test_every_key_generates_a_tile_of_the_right_size():
    pygame.init()
    for key in TEXTURE_KEYS:
        assert generate_texture(key).get_size() == (SIZE, SIZE), key
    assert set(DEFAULT_TEXTURE_KEYS) <= set(TEXTURE_KEYS)


def test_generation_is_deterministic_per_seed():
    pygame.init()
    a, b = generate_texture("wall", 7), generate_texture("wall", 7)
    assert _bytes(a) == _bytes(b)
    assert _bytes(generate_texture("wall", 8)) != _bytes(a)


def test_textures_look_like_their_subject():
    """Coarse colour sanity checks: grass is green, the sun is bright, ore is rock with rust veins, roofs are shingled."""
    pygame.init()
    ground = pygame.transform.average_color(generate_texture("ground"))
    assert ground[1] > ground[0] and ground[1] > ground[2]
    sun = generate_texture("sun")
    assert sun.get_at((SIZE // 2, SIZE // 2))[:3] >= (240, 240, 200) and sun.get_at((0, 0)).a == 0
    ore = generate_texture("ore")
    avg = pygame.transform.average_color(ore)
    assert avg[0] < 150 and avg[1] < 140 and avg[2] < 130                  # dark rock, slightly warm
    rusty = sum(1 for y in range(0, SIZE, 2) for x in range(0, SIZE, 2)
                if (c := ore.get_at((x, y)))[0] > 110 and c[0] > c[2] + 50)   # iron veins
    assert rusty > 40
    wall = generate_texture("wall")
    assert wall.get_at((10, 0))[:3] < (100, 100, 100)                    # mortar line along the top row
    roof = generate_texture("roof")
    assert roof.get_at((10, 0))[:3] < (100, 60, 40)                      # dark shingle edge


def test_unknown_key_is_magenta():
    assert generate_texture("does-not-exist").get_at((5, 5))[:3] == (255, 0, 255)
    assert placeholder_surface("x").get_at((5, 5))[:3] == (255, 0, 255)


def test_write_textures_overwrite_or_fill_gaps(tmp_path):
    pygame.init()
    written = write_textures(str(tmp_path), keys=("wall", "door"))
    assert sorted(os.path.basename(p) for p in written) == ["door.png", "wall.png"]
    custom = tmp_path / "wall.png"
    pygame.image.save(generate_texture("npc"), str(custom))
    before = custom.read_bytes()
    assert write_textures(str(tmp_path), keys=("wall", "ore"), overwrite=False) == [str(tmp_path / "ore.png")]
    assert custom.read_bytes() == before


def test_shipped_assets_cover_every_key():
    from unprepared.config import DEFAULT_ASSET_DIR
    for key in DEFAULT_TEXTURE_KEYS:
        assert os.path.exists(os.path.join(DEFAULT_ASSET_DIR, f"{key}.png")), key


def test_cli_regenerates_into_a_directory(tmp_path):
    import subprocess
    import sys
    from tests.conftest import PROJECT_DIR
    proc = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "tools", "textures", "gen_textures.py"),
                           "--out", str(tmp_path), "--only", "ore", "roof"],
                          capture_output=True, text=True, env=dict(os.environ, SDL_VIDEODRIVER="dummy"), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert sorted(os.listdir(tmp_path)) == ["ore.png", "roof.png"]


def test_a_corrupt_texture_file_is_a_placeholder_not_a_crash(tmp_path, monkeypatch):
    """Like a missing file, a PNG that cannot be decoded (truncated on disk, not an image at all)
    shows up magenta and is listed in ``missing`` - the game still starts. (GL upload stubbed.)"""
    import unprepared.rendering.textures as textures
    uploaded = []
    monkeypatch.setattr(textures, "upload_surface", lambda surface, mipmap=False, repeat=True, anisotropy=1: uploaded.append(surface) or len(uploaded))
    pygame.init()
    pygame.image.save(generate_texture("wall"), str(tmp_path / "wall.png"))
    (tmp_path / "door.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    (tmp_path / "roof.png").write_text("not an image")
    registry = textures.TextureRegistry(str(tmp_path), mipmaps=False)
    for key in ("wall", "door", "roof", "ore"):
        assert registry.get(key) > 0
    assert registry.missing == ["door", "roof", "ore"]
    assert _bytes(uploaded[0]) == _bytes(generate_texture("wall"))                # the good one as it is
    assert all(s.get_at((5, 5))[:3] == (255, 0, 255) for s in uploaded[1:])          # the rest magenta
