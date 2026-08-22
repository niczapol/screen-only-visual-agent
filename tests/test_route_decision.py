from vision_bot.route_decision import RouteAction, decide_route_action


def test_route_decision_moves_until_target_enters_tracking_radius():
    decision = decide_route_action(
        distance_to_target=2.0,
        ore_detected=False,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
    )

    assert decision.action is RouteAction.MOVE_TO_LOAD_RADIUS


def test_route_decision_marks_absent_only_inside_tracking_radius():
    decision = decide_route_action(
        distance_to_target=1.0,
        ore_detected=False,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
    )

    assert decision.action is RouteAction.MARK_ABSENT


def test_route_decision_moves_closer_to_confirm_dark_icons_inside_tracking_radius():
    decision = decide_route_action(
        distance_to_target=1.0,
        ore_detected=False,
        dark_ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
        dark_exclude_distance=0.25,
        dark_exclusion_enabled=True,
    )

    assert decision.action is RouteAction.MOVE_TO_NODE


def test_route_decision_ignores_dark_icons_when_exclusion_disabled():
    decision = decide_route_action(
        distance_to_target=0.2,
        ore_detected=False,
        dark_ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
        dark_exclude_distance=0.25,
        dark_exclusion_enabled=False,
    )

    assert decision.action is RouteAction.MARK_ABSENT


def test_route_decision_permanently_excludes_dark_icons_when_close():
    decision = decide_route_action(
        distance_to_target=0.2,
        ore_detected=False,
        dark_ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
        dark_exclude_distance=0.25,
        dark_exclusion_enabled=True,
    )

    assert decision.action is RouteAction.MARK_PERMANENT_EXCLUDED


def test_route_decision_moves_closer_when_ore_is_visible():
    decision = decide_route_action(
        distance_to_target=0.8,
        ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
    )

    assert decision.action is RouteAction.MOVE_TO_NODE


def test_route_decision_holds_at_visible_node_when_mining_is_disabled():
    decision = decide_route_action(
        distance_to_target=0.1,
        ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=False,
    )

    assert decision.action is RouteAction.HOLD_AT_NODE


def test_route_decision_reports_ready_to_mine_when_enabled_and_close():
    decision = decide_route_action(
        distance_to_target=0.1,
        ore_detected=True,
        tracking_radius=1.4,
        reached_distance=0.2,
        mining_enabled=True,
    )

    assert decision.action is RouteAction.READY_TO_MINE
