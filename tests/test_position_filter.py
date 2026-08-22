from vision_bot.position_filter import (
    CoordinateResyncState,
    choose_best_coord_candidate,
    choose_best_coord_candidate_with_resync,
    is_plausible_coord_update,
)


def test_position_filter_accepts_small_coord_update():
    assert is_plausible_coord_update(3722530600, 3731530100, max_jump=1.5)


def test_position_filter_rejects_large_ocr_jump():
    assert not is_plausible_coord_update(3737530800, 4737530600, max_jump=1.5)


def test_position_filter_selects_nearest_plausible_candidate():
    candidates = [3890538800, 3690538600, 5690538600]

    assert choose_best_coord_candidate(3697537600, candidates, max_jump=1.5) == 3690538600


def test_position_filter_selects_start_candidate_near_route_reference():
    candidates = [892539500, 3697543500, 5697543500]
    references = [3700525000]

    assert choose_best_coord_candidate(None, candidates, max_jump=1.5, reference_coords=references) == 3697543500


def test_position_filter_keeps_first_start_candidate_when_already_near_route():
    candidates = [5631564600, 5631364600, 2031204800, 887054000]
    references = [6330565000]

    assert choose_best_coord_candidate(None, candidates, max_jump=1.5, reference_coords=references) == 5631564600


def test_position_filter_keeps_first_ratchet_candidate_over_nearer_false_candidate():
    candidates = [6184395000, 6164595000, 164080000]
    references = [6264395000, 6134595000]

    assert choose_best_coord_candidate(None, candidates, max_jump=1.5, reference_coords=references) == 6184395000


def test_position_filter_prefers_route_reference_over_false_start_consensus():
    candidates = [3227622900, 5227622900, 3227022900, 8227622900]
    references = [5202631300, 5050631000, 5260593000]

    assert choose_best_coord_candidate(None, candidates, max_jump=1.5, reference_coords=references) == 5227622900


def test_position_filter_uses_axis_consensus_for_noisy_login_recovery_frame():
    candidates = [5562365300, 5582365900, 3382565500, 5582565500, 5582289200]

    assert choose_best_coord_candidate(None, candidates, max_jump=1.5) == 5582565500


def test_position_filter_resyncs_after_repeated_implausible_candidate():
    state = CoordinateResyncState()
    previous = 4620405800
    candidates = [4505315600]
    references = [4500310000]

    first, first_resynced = choose_best_coord_candidate_with_resync(
        previous,
        candidates,
        max_jump=1.5,
        reference_coords=references,
        resync_state=state,
        resync_confirm_steps=2,
    )
    second, second_resynced = choose_best_coord_candidate_with_resync(
        previous,
        candidates,
        max_jump=1.5,
        reference_coords=references,
        resync_state=state,
        resync_confirm_steps=2,
    )

    assert first is None
    assert not first_resynced
    assert second == 4505315600
    assert second_resynced


def test_position_filter_rejects_repeated_missing_leading_digit_resync():
    state = CoordinateResyncState()
    previous = 5076741400
    candidates = [88760200]

    for _ in range(4):
        selected, resynced = choose_best_coord_candidate_with_resync(
            previous,
            candidates,
            max_jump=4.83,
            reference_coords=[4859742400],
            resync_state=state,
            resync_confirm_steps=3,
            max_resync_distance=5.0,
        )
        assert selected is None
        assert not resynced


def test_position_filter_allows_confirmed_nearby_runtime_resync():
    state = CoordinateResyncState()
    previous = 5000700000
    candidates = [5250710000]

    first, _ = choose_best_coord_candidate_with_resync(
        previous,
        candidates,
        max_jump=1.5,
        reference_coords=[5250710000],
        resync_state=state,
        resync_confirm_steps=2,
        max_resync_distance=5.0,
    )
    second, resynced = choose_best_coord_candidate_with_resync(
        previous,
        candidates,
        max_jump=1.5,
        reference_coords=[5250710000],
        resync_state=state,
        resync_confirm_steps=2,
        max_resync_distance=5.0,
    )

    assert first is None
    assert second == 5250710000
    assert resynced
