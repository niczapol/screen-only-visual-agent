from __future__ import annotations

import math

import pytest

from vision_bot.core.evidence import Evidence, EvidenceState
from vision_bot.core.geometry import MapPoint, PhysicalRoute, ZoneGeometry


def test_evidence_keeps_absent_separate_from_unknown() -> None:
    absent = Evidence[bool].absent(
        observed_at=10.0,
        frame_id=3,
        source="combat_marker",
        reason="roi_read_and_marker_not_present",
    )
    unknown = Evidence[bool].unknown(
        observed_at=10.0,
        frame_id=3,
        source="combat_marker",
        reason="window_not_foreground",
    )

    assert absent.state is EvidenceState.ABSENT
    assert unknown.state is EvidenceState.UNKNOWN
    assert absent.present_value(now=10.1, max_age_seconds=0.5) is None
    assert unknown.present_value(now=10.1, max_age_seconds=0.5) is None


def test_evidence_rejects_present_without_value() -> None:
    with pytest.raises(ValueError, match="present evidence requires"):
        Evidence(
            state=EvidenceState.PRESENT,
            value=None,
            confidence=1.0,
            observed_at=1.0,
            frame_id=1,
            source="test",
        )


def test_zone_geometry_uses_physical_axis_scale() -> None:
    tanaris = ZoneGeometry(zone_id=440, width_yards=6900.0, height_yards=4600.0)
    origin = MapPoint(50.0, 50.0)

    assert tanaris.distance(origin, MapPoint(51.0, 50.0)) == pytest.approx(69.0)
    assert tanaris.distance(origin, MapPoint(50.0, 51.0)) == pytest.approx(46.0)
    assert tanaris.heading_degrees(origin, MapPoint(51.0, 51.0)) == pytest.approx(
        236.309932,
        abs=1.0e-5,
    )


def test_physical_route_lookahead_is_measured_in_yards() -> None:
    geometry = ZoneGeometry(zone_id=1, width_yards=1000.0, height_yards=500.0)
    route = PhysicalRoute(
        [MapPoint(0.0, 0.0), MapPoint(10.0, 0.0), MapPoint(10.0, 10.0)],
        geometry,
        loop=False,
    )

    assert route.total_length_yards == pytest.approx(150.0)
    assert route.point_at(125.0) == MapPoint(10.0, 5.0)
    projection = route.project(MapPoint(9.0, 6.0))
    assert projection.segment_index == 1
    assert projection.distance_along_yards == pytest.approx(130.0)
    assert projection.cross_track_yards == pytest.approx(10.0)
    assert math.isclose(projection.segment_fraction, 0.6)
