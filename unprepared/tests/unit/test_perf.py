"""Frame profiler, quality recommendation and graphics settings."""
import json

import pytest

from unprepared.graphics import PRESETS, GraphicsSettings, graphics_settings_path
from unprepared.perf import FrameProfiler, is_software_renderer, recommend_quality
from unprepared.ui import Menu, MenuItem, slider_text


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestFrameProfiler:
    def test_phases_and_frame_times(self):
        clock = FakeClock()
        p = FrameProfiler(history=4, clock=clock)
        with p.phase("update"):
            clock.advance(0.004)
        with p.phase("render3d"):
            clock.advance(0.010)
        clock.advance(0.002)
        total = p.end_frame()
        assert total == pytest.approx(16.0)
        assert p.last_phase_ms == {"update": pytest.approx(4.0), "render3d": pytest.approx(10.0)}
        assert p.frames == 1 and p.fps == pytest.approx(62.5)
        stats = p.stats()
        assert stats["frame_ms"] == 16.0 and stats["phases"]["render3d"] == 10.0

    def test_history_is_bounded_and_worst_tracked(self):
        clock = FakeClock()
        p = FrameProfiler(history=3, clock=clock)
        for ms in (10, 50, 20, 30):
            clock.advance(ms / 1000.0)
            p.end_frame()
        assert list(p.frame_history) == pytest.approx([50.0, 20.0, 30.0])
        assert p.worst_frame_ms == pytest.approx(50.0) and p.average_frame_ms == pytest.approx(100 / 3)

    def test_nested_phases_accumulate(self):
        clock = FakeClock()
        p = FrameProfiler(clock=clock)
        for _ in range(2):
            with p.phase("hud"):
                clock.advance(0.001)
        p.end_frame()
        assert p.last_phase_ms["hud"] == pytest.approx(2.0)


class TestRecommendation:
    def test_by_frame_time(self):
        assert recommend_quality(6.0, "NVIDIA GeForce RTX") == "high"
        assert recommend_quality(15.0, "Intel UHD") == "medium"
        assert recommend_quality(40.0, "Intel UHD") == "low"

    def test_software_renderers_go_low(self):
        assert is_software_renderer("llvmpipe (LLVM 20.1.2, 256 bits)")
        assert not is_software_renderer("AMD Radeon RX 6700")
        assert recommend_quality(3.0, "Mesa llvmpipe") == "low"
        # every renderer the docs name as software: Chrome/ANGLE's SwiftShader among them
        for name in ("Google SwiftShader", "SwiftShader Device (Subzero) (0x0000C0DE)", "softpipe", "Mesa Offscreen"):
            assert is_software_renderer(name) and recommend_quality(3.0, name) == "low", name


class TestGraphicsSettings:
    def test_presets(self):
        g = GraphicsSettings()
        g.apply_preset("low")
        assert g.quality == "low" and not g.shaders and g.object_fade < 1.0 and g.star_count == 0
        g.apply_preset("high")
        assert g.shaders and g.star_count == PRESETS["high"]["star_count"]
        assert g.next_quality(+1) == "low" and not g.auto_quality           # wraps high -> low

    def test_draw_distances(self):
        g = GraphicsSettings(object_fade=0.5, character_fade=0.25)
        assert g.object_draw_distance(100.0) == 50.0 and g.character_draw_distance(100.0) == 25.0
        assert GraphicsSettings(object_fade=5.0).object_draw_distance(100.0) == 100.0

    def test_roundtrip_and_unknown_fields(self, tmp_path):
        g = GraphicsSettings(fov=75.0, contrast=1.2, quality="medium")
        path = graphics_settings_path(str(tmp_path / "savegame.json"))
        assert path == str(tmp_path / "graphics.json")
        g.save(path)
        loaded = GraphicsSettings.load(path)
        assert loaded.fov == 75.0 and loaded.contrast == 1.2 and loaded.quality == "medium"
        with open(path, "w") as f:
            json.dump({"fov": 66.0, "bogus": 1}, f)
        assert GraphicsSettings.load(path).fov == 66.0
        with open(path, "w") as f:
            f.write("not json")
        assert GraphicsSettings.load(path) is None
        assert GraphicsSettings.load(str(tmp_path / "missing.json")) is None


    @pytest.mark.parametrize("content", ["[1, 2]", '"high"', "null", "3"])
    def test_json_that_is_not_an_object_is_ignored_like_a_corrupt_file(self, tmp_path, content):
        """The game loads this before its window exists: a bad file means "start afresh", not a crash."""
        path = tmp_path / "graphics.json"
        path.write_text(content)
        assert GraphicsSettings.load(str(path)) is None

    def test_an_unknown_preset_or_a_value_of_the_wrong_type_falls_back_to_the_default(self, tmp_path):
        path = tmp_path / "graphics.json"
        path.write_text(json.dumps({"quality": "ultra", "fov": "wide", "star_count": 12.5, "shaders": 1,
                                    "contrast": 1, "roofs": False}))
        g = GraphicsSettings.load(str(path))
        default = GraphicsSettings()
        assert g.quality == default.quality and g.fov == default.fov and g.star_count == default.star_count
        assert g.shaders is default.shaders                       # 1 is not a boolean
        assert g.contrast == 1.0 and isinstance(g.contrast, float) and g.roofs is False   # the good ones are kept
        assert g.next_quality(+1) in ("low", "medium", "high")    # the options menu can step on from it

    def test_a_settings_file_that_cannot_be_written_is_not_fatal(self, tmp_path, monkeypatch):
        import unprepared.graphics as graphics

        def read_only(*args, **kwargs):
            raise PermissionError("read-only file system")
        monkeypatch.setattr(graphics, "write_json", read_only)
        assert GraphicsSettings().save(str(tmp_path / "graphics.json")) is False
        monkeypatch.undo()
        assert GraphicsSettings().save(str(tmp_path / "graphics.json")) is True


class TestMenu:
    def test_navigation_and_actions(self):
        hits = []
        value = {"fov": 60.0}
        menu = Menu("T", [
            MenuItem("Go", lambda: hits.append("go")),
            MenuItem("FOV", adjust=lambda d: value.__setitem__("fov", value["fov"] + 2 * d), value=lambda: str(value["fov"])),
            MenuItem("Off", lambda: hits.append("off"), enabled=lambda: False),
        ])
        assert menu.current.label == "Go"
        menu.activate()
        assert hits == ["go"]
        assert menu.move(-1) == 2
        menu.activate()
        assert hits == ["go"]                     # disabled
        menu.move(-1)
        menu.adjust(+1)
        menu.adjust(+1)
        menu.adjust(-1)
        assert value["fov"] == 62.0 and menu.current.display_value() == "62.0"
        menu.activate()                            # Enter on an adjustable item nudges it up
        assert value["fov"] == 64.0

    def test_slider_text(self):
        assert slider_text(60, 50, 100, width=10, suffix=" deg") == "[##........] 60 deg"
        assert slider_text(1.0, 0.3, 1.0, width=4) == "[####] 1"
        assert slider_text(5, 5, 5) .startswith("[")
