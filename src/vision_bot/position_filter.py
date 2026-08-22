from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vision_bot.coords import coord_to_values
from vision_bot.route_planner import MiningRoutePlanner


@dataclass
class CoordinateResyncState:
    candidate: int | None = None
    count: int = 0


def is_plausible_coord_update(previous_coord: int | None, current_coord: int, max_jump: float) -> bool:
    if previous_coord is None:
        return True
    if max_jump <= 0:
        return True
    return MiningRoutePlanner.distance(previous_coord, current_coord) <= max_jump


def choose_best_coord_candidate(
    previous_coord: int | None,
    candidates: list[int],
    max_jump: float,
    reference_coords: Iterable[int] | None = None,
    start_reference_trust_radius: float = 5.0,
) -> int | None:
    if not candidates:
        return None
    if previous_coord is None:
        references = list(reference_coords or [])
        if references:
            first_candidate = candidates[0]
            if _nearest_reference_distance(first_candidate, references) <= max(0.0, start_reference_trust_radius):
                return first_candidate
            return min(candidates, key=lambda candidate: _nearest_reference_distance(candidate, references))
        consensus_candidate = _consensus_start_candidate(candidates)
        if consensus_candidate is not None:
            return consensus_candidate
        return candidates[0]

    plausible = [candidate for candidate in candidates if is_plausible_coord_update(previous_coord, candidate, max_jump)]
    if not plausible:
        return None

    return min(plausible, key=lambda candidate: MiningRoutePlanner.distance(previous_coord, candidate))


def choose_best_coord_candidate_with_resync(
    previous_coord: int | None,
    candidates: list[int],
    max_jump: float,
    reference_coords: Iterable[int] | None,
    resync_state: CoordinateResyncState,
    resync_confirm_steps: int = 3,
    max_resync_distance: float | None = None,
) -> tuple[int | None, bool]:
    selected = choose_best_coord_candidate(previous_coord, candidates, max_jump, reference_coords)
    if selected is not None:
        resync_state.candidate = None
        resync_state.count = 0
        return selected, False

    if not candidates:
        resync_state.candidate = None
        resync_state.count = 0
        return None, False

    resync_candidate = choose_best_coord_candidate(None, candidates, max_jump, reference_coords)
    if resync_candidate is None:
        return None, False
    if (
        previous_coord is not None
        and max_resync_distance is not None
        and max_resync_distance > 0.0
        and MiningRoutePlanner.distance(previous_coord, resync_candidate) > max_resync_distance
    ):
        resync_state.candidate = None
        resync_state.count = 0
        return None, False

    if resync_state.candidate == resync_candidate:
        resync_state.count += 1
    else:
        resync_state.candidate = resync_candidate
        resync_state.count = 1

    if resync_state.count >= max(1, resync_confirm_steps):
        resync_state.candidate = None
        resync_state.count = 0
        return resync_candidate, True

    return None, False


def _nearest_reference_distance(coord: int, reference_coords: list[int]) -> float:
    return min(MiningRoutePlanner.distance(coord, reference_coord) for reference_coord in reference_coords)


def _consensus_start_candidate(candidates: list[int]) -> int | None:
    if len(candidates) < 3:
        return None

    x_counts: dict[int, int] = {}
    y_counts: dict[int, int] = {}
    for candidate in candidates:
        x_value, y_value = _decode_coord_values(candidate)
        x_counts[x_value] = x_counts.get(x_value, 0) + 1
        y_counts[y_value] = y_counts.get(y_value, 0) + 1

    x_value = _unique_plurality_value(x_counts)
    y_value = _unique_plurality_value(y_counts)
    if x_value is None or y_value is None:
        return None

    consensus = int(f"{x_value:04d}{y_value:04d}00")
    if consensus in candidates:
        return consensus

    nearest = min(candidates, key=lambda candidate: MiningRoutePlanner.distance(candidate, consensus))
    if MiningRoutePlanner.distance(nearest, consensus) <= 0.35:
        return consensus
    return None


def _unique_plurality_value(counts: dict[int, int]) -> int | None:
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] < 2:
        return None
    if len(ordered) > 1 and ordered[1][1] == ordered[0][1]:
        return None
    return ordered[0][0]


def _decode_coord_values(coord: int) -> tuple[int, int]:
    return coord_to_values(coord)
