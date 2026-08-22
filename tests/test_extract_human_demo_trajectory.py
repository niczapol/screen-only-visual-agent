from scripts.extract_human_demo_trajectory import (
    CoordinateCandidate,
    coordinate_candidates_from_text,
    interpolate_trajectory,
    select_continuous_candidates,
)


def test_coordinate_candidates_parse_normal_and_separator_free_ocr():
    normal = coordinate_candidates_from_text("49.27, 25.50", source="gray")
    digits = coordinate_candidates_from_text("49272550", source="gray")

    assert (normal[0].x, normal[0].y) == (49.27, 25.5)
    assert (digits[0].x, digits[0].y) == (49.27, 25.5)
    assert not normal[0].repaired


def test_coordinate_candidates_offer_bounded_missing_digit_repairs():
    candidates = coordinate_candidates_from_text(
        "9272550",
        source="gray",
        bounds=(40.0, 60.0, 20.0, 30.0),
    )

    assert any((candidate.x, candidate.y) == (49.27, 25.5) for candidate in candidates)
    assert all(candidate.repaired for candidate in candidates)


def test_temporal_selection_rejects_cheaper_impossible_jump():
    observations = [
        (0.0, [CoordinateCandidate(49.0, 25.0, 0.0, "gray")]),
        (
            1.0,
            [
                CoordinateCandidate(9.1, 25.1, 0.0, "gray"),
                CoordinateCandidate(49.1, 25.1, 3.0, "gray:insert", True),
            ],
        ),
        (2.0, [CoordinateCandidate(49.2, 25.2, 0.0, "gray")]),
    ]

    selected = select_continuous_candidates(observations, max_speed=1.5)

    assert [(candidate.x, candidate.y) for candidate in selected if candidate] == [
        (49.0, 25.0),
        (49.1, 25.1),
        (49.2, 25.2),
    ]


def test_temporal_selection_can_skip_only_impossible_candidate():
    observations = [
        (0.0, [CoordinateCandidate(49.0, 25.0, 0.0, "gray")]),
        (1.0, [CoordinateCandidate(9.0, 80.0, 0.0, "gray")]),
        (2.0, [CoordinateCandidate(49.2, 25.2, 0.0, "gray")]),
    ]

    selected = select_continuous_candidates(observations, max_speed=1.5)

    assert selected[0] is not None
    assert selected[1] is None
    assert selected[2] is not None


def test_interpolation_fills_only_short_gaps():
    selected = [
        CoordinateCandidate(10.0, 20.0, 0.0, "gray"),
        None,
        CoordinateCandidate(12.0, 22.0, 0.0, "gray"),
    ]

    filled = interpolate_trajectory([0.0, 1.0, 2.0], selected, max_gap=3.0)

    assert filled[1] == (11.0, 21.0, True)
