from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vision_bot.route_probe_observation import observe_route_frame


def test_observe_route_frame_returns_immutable_disabled_observation():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    observation = observe_route_frame(
        frame,
        {},
        timestamp=12.5,
        recognize_ore=False,
    )

    assert observation.timestamp == 12.5
    assert observation.telemetry is None
    assert observation.bright_ore_points == ()
    assert observation.dark_ore_points == ()
    assert not observation.armor_critical
    assert not observation.mounted
    assert not observation.forbidden_subzone_visible
    with pytest.raises(FrozenInstanceError):
        observation.mounted = True


def test_observe_route_frame_normalizes_ore_points_to_tuples(monkeypatch):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    minimap = np.zeros((40, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "vision_bot.route_probe_observation.recognize_ore_point_classes",
        lambda _image, _config: ([(3, 4)], [(8, 9)]),
    )

    observation = observe_route_frame(
        frame,
        {},
        timestamp=3.0,
        minimap=minimap,
        recognize_ore=True,
    )

    assert observation.bright_ore_points == ((3, 4),)
    assert observation.dark_ore_points == ((8, 9),)
