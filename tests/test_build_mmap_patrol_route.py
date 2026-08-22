from __future__ import annotations

from scripts.build_mmap_patrol_route import farthest_reachable_location
from vision_bot.mmap_navmesh import MMapNavMesh, NavLocation, NavPolygon


def test_farthest_reachable_location_respects_blocked_edges() -> None:
    mesh = MMapNavMesh()
    mesh.polygons = {
        0: _polygon(0, 0.0),
        1: _polygon(1, 2.0),
        2: _polygon(2, 4.0),
        3: _polygon(3, -3.0),
    }
    _connect(mesh, 0, 1)
    _connect(mesh, 1, 2)
    _connect(mesh, 0, 3)

    target, count = farthest_reachable_location(
        mesh,
        NavLocation(0, mesh.polygons[0].center, 0.0, 0.0),
        transition_cost=lambda first, second, _portal: (
            None if {first.polygon_id, second.polygon_id} == {1, 2} else 1.0
        ),
        max_graph_cost=10.0,
    )

    assert count == 3
    assert target is not None
    assert target.polygon_id == 3


def test_farthest_reachable_location_respects_cost_limit() -> None:
    mesh = MMapNavMesh()
    mesh.polygons = {0: _polygon(0, 0.0), 1: _polygon(1, 2.0), 2: _polygon(2, 4.0)}
    _connect(mesh, 0, 1)
    _connect(mesh, 1, 2)

    target, count = farthest_reachable_location(
        mesh,
        NavLocation(0, mesh.polygons[0].center, 0.0, 0.0),
        transition_cost=lambda *_args: 2.0,
        max_graph_cost=2.5,
    )

    assert count == 2
    assert target is not None
    assert target.polygon_id == 1


def _polygon(polygon_id: int, x: float) -> NavPolygon:
    return NavPolygon(
        polygon_id,
        32,
        32,
        polygon_id,
        ((x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 0.0, 1.0)),
        1,
        1,
    )


def _connect(mesh: MMapNavMesh, first: int, second: int) -> None:
    portal = ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    mesh.polygons[first].neighbors[second] = portal
    mesh.polygons[second].neighbors[first] = portal[::-1]
