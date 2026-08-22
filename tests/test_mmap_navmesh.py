from __future__ import annotations

import math
from pathlib import Path

import pytest

from vision_bot.mmap_navmesh import (
    NavLocation,
    NavPolygon,
    MMapNavMesh,
    _funnel_path,
    densify_path,
    mmtile_path,
    world_to_tile,
)


def test_world_to_tile_matches_current_desolace_position() -> None:
    assert world_to_tile(1549.0355, -697.6492) == (33, 29)


def test_mmtile_path_uses_azerothcore_swapped_file_axes(tmp_path: Path) -> None:
    assert mmtile_path(tmp_path, map_id=1, tile_x=33, tile_y=29) == (
        tmp_path / "0012933.mmtile"
    )


def test_funnel_path_stays_straight_through_aligned_portals() -> None:
    start = (0.0, 0.0, 0.0)
    end = (10.0, 0.0, 0.0)
    portals = [
        ((3.0, 0.0, 2.0), (3.0, 0.0, -2.0)),
        ((7.0, 0.0, 2.0), (7.0, 0.0, -2.0)),
    ]
    centers = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]

    assert _funnel_path(start, end, portals, centers) == [start, end]


def test_densify_path_caps_horizontal_spacing() -> None:
    points = densify_path([(0.0, 0.0, 0.0), (10.0, 5.0, 0.0)], 3.0)

    assert len(points) == 5
    assert points[-1] == (10.0, 5.0, 0.0)
    assert max(
        math.dist((first[0], first[2]), (second[0], second[2]))
        for first, second in zip(points, points[1:])
    ) <= 3.0


def test_find_path_uses_polygon_portal() -> None:
    mesh = MMapNavMesh()
    first = NavPolygon(
        polygon_id=0,
        tile_x=32,
        tile_y=32,
        local_index=0,
        vertices=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 5.0), (0.0, 0.0, 5.0)),
        flags=1,
        area=1,
    )
    second = NavPolygon(
        polygon_id=1,
        tile_x=32,
        tile_y=32,
        local_index=1,
        vertices=((5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 5.0), (5.0, 0.0, 5.0)),
        flags=1,
        area=1,
    )
    portal = ((5.0, 0.0, 0.0), (5.0, 0.0, 5.0))
    first.neighbors[1] = portal
    second.neighbors[0] = (portal[1], portal[0])
    mesh.polygons = {0: first, 1: second}

    path = mesh.find_path(
        NavLocation(0, (1.0, 0.0, 2.5), 0.0, 0.0),
        NavLocation(1, (9.0, 0.0, 2.5), 0.0, 0.0),
    )

    assert path is not None
    assert path.polygon_ids == (0, 1)
    assert path.points == ((1.0, 0.0, 2.5), (9.0, 0.0, 2.5))
    assert path.world_length == pytest.approx(8.0)


def test_find_path_can_reject_terrain_blocked_transition() -> None:
    mesh = MMapNavMesh()
    first = NavPolygon(0, 32, 32, 0, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0)), 1, 1)
    blocked = NavPolygon(1, 32, 32, 1, ((2.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 0.0, 2.0)), 1, 1)
    detour = NavPolygon(2, 32, 32, 2, ((0.0, 0.0, 2.0), (2.0, 0.0, 2.0), (2.0, 0.0, 4.0)), 1, 1)
    goal = NavPolygon(3, 32, 32, 3, ((2.0, 0.0, 2.0), (4.0, 0.0, 2.0), (4.0, 0.0, 4.0)), 1, 1)
    first.neighbors = {
        1: ((2.0, 0.0, 0.0), (2.0, 0.0, 2.0)),
        2: ((0.0, 0.0, 2.0), (2.0, 0.0, 2.0)),
    }
    blocked.neighbors = {0: first.neighbors[1][::-1], 3: ((2.0, 0.0, 2.0), (4.0, 0.0, 2.0))}
    detour.neighbors = {0: first.neighbors[2][::-1], 3: ((2.0, 0.0, 2.0), (2.0, 0.0, 4.0))}
    goal.neighbors = {1: blocked.neighbors[3][::-1], 2: detour.neighbors[3][::-1]}
    mesh.polygons = {0: first, 1: blocked, 2: detour, 3: goal}

    path = mesh.find_path(
        NavLocation(0, (1.0, 0.0, 1.0), 0.0, 0.0),
        NavLocation(3, (3.0, 0.0, 3.0), 0.0, 0.0),
        transition_cost=lambda current, neighbor, _portal: (
            None if {current.polygon_id, neighbor.polygon_id} == {0, 1} else 1.0
        ),
    )

    assert path is not None
    assert path.polygon_ids == (0, 2, 3)
