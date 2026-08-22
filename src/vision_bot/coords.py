from __future__ import annotations


def coord_to_xy(coord: int) -> tuple[float, float]:
    """Decode GatherMate-style packed coordinates into map percentages."""
    coord_str = str(int(coord))
    if len(coord_str) <= 8:
        values = coord_str.zfill(8)
        return int(values[:4]) / 100.0, int(values[4:8]) / 100.0

    values = coord_str.zfill(10)
    return int(values[:4]) / 100.0, int(values[4:8]) / 100.0


def coord_to_values(coord: int) -> tuple[int, int]:
    x, y = coord_to_xy(coord)
    return int(round(x * 100)), int(round(y * 100))


def xy_to_coord(x: float, y: float, *, style: str = "gathermate2") -> int:
    x_value = int(round(x * 100))
    y_value = int(round(y * 100))
    if style == "gathermate1":
        return int(f"{x_value:04d}{y_value:04d}")
    return int(f"{x_value:04d}{y_value:04d}00")


def normalize_coord(coord: int, *, style: str = "gathermate2") -> int:
    x, y = coord_to_xy(coord)
    return xy_to_coord(x, y, style=style)
