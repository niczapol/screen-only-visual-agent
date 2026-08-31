from __future__ import annotations

import numpy as np

from scripts.build_tanaris_rail_route_v19 import build_v19
from vision_bot.coords import xy_to_coord


class _Raster:
    @staticmethod
    def ui_to_grid(x: float, y: float, zone: object) -> tuple[float, float]:
        return y, x


class _Planner:
    zone = object()
    raster = _Raster()
    hazard_mask = np.zeros((101, 101), dtype=bool)

    @staticmethod
    def coord_is_hazard(coord: int) -> bool:
        return False


def test_v19_replaces_unsafe_excursion_and_remaps_safe_access_plan() -> None:
    route = [
        {"index": index, "coord": xy_to_coord(float(index), 10.0)}
        for index in range(4)
    ]
    source = {
        "schema_version": 1,
        "zone_id": 162,
        "name": "v18 fixture",
        "route_loop": route,
        "route_nodes": [
            {
                "node_id": 7,
                "coord": xy_to_coord(3.0, 10.0),
                "ore_type": "Iron",
                "terrain_access_plan": {
                    "attachment_route_index": 0,
                    "resume_route_index": 3,
                    "primary_option_rank": 0,
                    "options": [
                        {
                            "rank": 0,
                            "primary": True,
                            "inbound": {"coords": [xy_to_coord(0.0, 10.0)]},
                            "return": {"coords": [xy_to_coord(3.0, 10.0)]},
                            "resume": {"coords": [xy_to_coord(3.0, 10.0)]},
                        }
                    ],
                },
            }
        ],
        "rejected_nodes": [],
        "metrics": {},
        "route_override": {"reason": "historical"},
    }

    built = build_v19(
        source,
        planner=_Planner(),
        bridge_coords=(xy_to_coord(1.5, 9.0),),
        bridge_validation={"walkable": True},
        source_fingerprint="fixture-sha",
        remove_start=1,
        remove_end=2,
    )

    assert [item["coord"] for item in built["route_loop"]] == [
        route[0]["coord"],
        xy_to_coord(1.5, 9.0),
        route[3]["coord"],
    ]
    plan = built["route_nodes"][0]["terrain_access_plan"]
    assert plan["attachment_route_index"] == 0
    assert plan["resume_route_index"] == 2
    assert built["source"]["source_sha256"] == "fixture-sha"
    assert built["route_override"]["all_hazard_audit"] == {
        "hazard_point_indexes": [],
        "hazard_segment_indexes": [],
    }
