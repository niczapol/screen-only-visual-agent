from dataclasses import replace

import pytest

from vision_bot.node_access import NodeAccessController, NodeAccessState
from vision_bot.route_planner import NodeAccessOption, NodeAccessPlan


def _option(
    rank: int,
    *,
    inbound: tuple[int, ...],
    resume: tuple[int, ...],
    primary: bool = False,
) -> NodeAccessOption:
    return NodeAccessOption(
        rank=rank,
        candidate_index=rank,
        primary=primary,
        approach_coord=inbound[-1],
        sector=rank,
        bearing_degrees=rank * 45.0,
        objective=float(rank),
        inbound_coords=inbound,
        return_coords=tuple(reversed(inbound)),
        resume_coords=resume,
    )


def _plan(*options: NodeAccessOption, primary_rank: int = 0) -> NodeAccessPlan:
    return NodeAccessPlan(
        attachment_route_index=4,
        resume_route_index=6,
        primary_option_rank=primary_rank,
        options=options,
    )


def test_primary_access_reaches_ore_check_then_resumes_route():
    controller = NodeAccessController()
    controller.start(
        _plan(_option(0, inbound=(10, 11, 12), resume=(12, 13, 14), primary=True))
    )

    assert controller.state == NodeAccessState.INBOUND
    assert controller.current_target_coord == 10
    for expected in (11, 12):
        controller.mark_target_reached()
        assert controller.current_target_coord == expected
    controller.mark_target_reached()

    assert controller.state == NodeAccessState.ORE_CHECK
    assert controller.current_target_coord is None
    controller.resume_after_ore_check()
    for expected in (12, 13, 14):
        assert controller.current_target_coord == expected
        controller.mark_target_reached()

    assert controller.state == NodeAccessState.COMPLETE
    assert controller.resume_route_index == 6


def test_blocked_access_backtracks_only_reached_breadcrumb_then_tries_fallback():
    primary = _option(0, inbound=(10, 11, 12, 13), resume=(13, 14), primary=True)
    fallback = _option(1, inbound=(10, 21, 22), resume=(22, 14))
    controller = NodeAccessController()
    controller.start(_plan(fallback, primary))

    controller.mark_target_reached()
    controller.mark_target_reached()
    controller.mark_access_blocked()

    assert controller.state == NodeAccessState.BACKTRACK
    assert controller.current_target_coord == 11
    controller.mark_target_reached()
    assert controller.current_target_coord == 10
    controller.mark_target_reached()

    assert controller.state == NodeAccessState.INBOUND
    assert controller.current_option == fallback
    assert controller.current_target_coord == 10
    assert controller.attempted_option_count == 2


def test_absent_ore_resumes_without_trying_another_side():
    primary = _option(0, inbound=(10, 12), resume=(12, 14), primary=True)
    fallback = _option(1, inbound=(10, 22), resume=(22, 14))
    controller = NodeAccessController()
    controller.start(_plan(primary, fallback))
    controller.mark_target_reached()
    controller.mark_target_reached()

    controller.resume_after_ore_check()

    assert controller.state == NodeAccessState.RESUME
    assert controller.current_option == primary
    assert controller.attempted_option_count == 1


def test_all_blocked_options_exhaust_without_permanent_exclusion_decision():
    controller = NodeAccessController()
    controller.start(
        _plan(
            _option(0, inbound=(10, 12), resume=(12, 14), primary=True),
            _option(1, inbound=(10, 22), resume=(22, 14)),
        )
    )

    controller.mark_access_blocked()
    assert controller.current_option is not None
    assert controller.current_option.rank == 1
    controller.mark_access_blocked()

    assert controller.state == NodeAccessState.EXHAUSTED
    assert controller.current_target_coord is None
    assert controller.resume_route_index == 6
    assert controller.attempted_option_count == 2


def test_invalid_state_transitions_fail_closed():
    controller = NodeAccessController()
    with pytest.raises(RuntimeError):
        controller.mark_target_reached()
    with pytest.raises(RuntimeError):
        controller.mark_access_blocked()
    with pytest.raises(RuntimeError):
        controller.resume_after_ore_check()


def test_start_rejects_return_path_that_is_not_reverse_breadcrumb():
    option = _option(0, inbound=(10, 11, 12), resume=(12, 14), primary=True)
    invalid = replace(option, return_coords=(12, 10))

    with pytest.raises(ValueError, match="reverse"):
        NodeAccessController().start(_plan(invalid))
