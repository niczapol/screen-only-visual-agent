"""Run a deterministic route-following replay without a game client or input."""

from __future__ import annotations

import json

from vision_bot.coords import coord_to_xy, xy_to_coord
from vision_bot.route_following import DirectedRouteFollower


def main() -> None:
    route = [
        xy_to_coord(10.0, 10.0),
        xy_to_coord(20.0, 10.0),
        xy_to_coord(20.0, 20.0),
        xy_to_coord(10.0, 20.0),
    ]
    replay = [
        xy_to_coord(11.0, 10.1),
        xy_to_coord(15.0, 9.9),
        xy_to_coord(19.8, 10.2),
        xy_to_coord(20.1, 15.0),
        # A backward-jitter sample must not regress accepted progress.
        xy_to_coord(20.0, 14.8),
        # A far-off sample must be rejected without corrupting state.
        xy_to_coord(50.0, 50.0),
        xy_to_coord(19.9, 19.8),
    ]
    follower = DirectedRouteFollower(
        route,
        lookahead_distance=2.0,
        corridor_radius=1.0,
        backward_tolerance=0.30,
        max_forward_advance=2.0,
        pass_tolerance=0.05,
    )

    rows = []
    for index, coordinate in enumerate(replay):
        observation = follower.observe(coordinate)
        rows.append(
            {
                "frame": index,
                "observed_xy": coord_to_xy(coordinate),
                "accepted_progress": round(observation.progress, 3),
                "progress_delta": round(observation.progress_delta, 3),
                "cross_track": round(observation.cross_track_distance, 3),
                "accepted": observation.projection_accepted,
                "steering_xy": coord_to_xy(observation.steering_coord),
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

