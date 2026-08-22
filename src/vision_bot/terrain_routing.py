from __future__ import annotations

import heapq
import math
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import cv2
import numpy as np


TILE_SIZE = 533.3333333333
TILE_GRID_CELLS = 128
V9_SIZE = 129
GRID_SPACING = TILE_SIZE / TILE_GRID_CELLS

GridPoint = tuple[int, int]


@dataclass(frozen=True)
class TerrainCostConfig:
    soft_slope_degrees: float = 15.0
    hard_slope_degrees: float = 35.0
    slope_weight: float = 8.0
    roughness_weight: float = 0.08
    clearance_cells: float = 3.0
    clearance_weight: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.soft_slope_degrees < self.hard_slope_degrees < 90.0:
            raise ValueError("Expected 0 <= soft slope < hard slope < 90")
        if self.slope_weight < 0.0 or self.roughness_weight < 0.0:
            raise ValueError("Terrain cost weights must be non-negative")
        if self.clearance_cells < 0.0 or self.clearance_weight < 0.0:
            raise ValueError("Clearance settings must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ZoneTransform:
    left: float
    right: float
    top: float
    bottom: float

    def __post_init__(self) -> None:
        if math.isclose(self.left, self.right) or math.isclose(self.top, self.bottom):
            raise ValueError("Zone bounds must span non-zero width and height")

    def ui_to_world(self, ui_x: float, ui_y: float) -> tuple[float, float]:
        # WorldMapArea stores y1/y2 in fields historically named left/right
        # and x1/x2 in fields named top/bottom. Client map axes are swapped.
        world_x = self.top + (self.bottom - self.top) * (float(ui_y) / 100.0)
        world_y = self.left + (self.right - self.left) * (float(ui_x) / 100.0)
        return world_x, world_y

    def world_to_ui(self, world_x: float, world_y: float) -> tuple[float, float]:
        ui_x = (float(world_y) - self.left) / (self.right - self.left) * 100.0
        ui_y = (float(world_x) - self.top) / (self.bottom - self.top) * 100.0
        return ui_x, ui_y


@dataclass(frozen=True)
class TerrainRaster:
    heights: np.ndarray
    tile_min_x: int
    tile_min_y: int
    tile_max_x: int
    tile_max_y: int

    def __post_init__(self) -> None:
        if self.tile_min_x > self.tile_max_x or self.tile_min_y > self.tile_max_y:
            raise ValueError("Invalid terrain tile bounds")
        expected_shape = (
            (self.tile_max_y - self.tile_min_y + 1) * TILE_GRID_CELLS + 1,
            (self.tile_max_x - self.tile_min_x + 1) * TILE_GRID_CELLS + 1,
        )
        if self.heights.shape != expected_shape:
            raise ValueError(f"Terrain raster shape {self.heights.shape} != expected {expected_shape}")
        if self.heights.ndim != 2:
            raise ValueError("Terrain heights must be a 2D array")

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.heights)

    def ui_to_grid(self, ui_x: float, ui_y: float, zone: ZoneTransform) -> tuple[float, float]:
        world_x, world_y = zone.ui_to_world(ui_x, ui_y)
        tile_float_x = 32.0 - world_y / TILE_SIZE
        tile_float_y = 32.0 - world_x / TILE_SIZE
        col = (tile_float_x - self.tile_min_x) * TILE_GRID_CELLS
        row = (tile_float_y - self.tile_min_y) * TILE_GRID_CELLS
        return row, col

    def grid_to_ui(self, row: float, col: float, zone: ZoneTransform) -> tuple[float, float]:
        tile_float_x = self.tile_min_x + float(col) / TILE_GRID_CELLS
        tile_float_y = self.tile_min_y + float(row) / TILE_GRID_CELLS
        world_y = (32.0 - tile_float_x) * TILE_SIZE
        world_x = (32.0 - tile_float_y) * TILE_SIZE
        return zone.world_to_ui(world_x, world_y)

    def sample_height(self, row: float, col: float) -> float | None:
        if not (0.0 <= row <= self.heights.shape[0] - 1 and 0.0 <= col <= self.heights.shape[1] - 1):
            return None
        row0 = int(math.floor(row))
        col0 = int(math.floor(col))
        row1 = min(row0 + 1, self.heights.shape[0] - 1)
        col1 = min(col0 + 1, self.heights.shape[1] - 1)
        values = np.array(
            [
                self.heights[row0, col0],
                self.heights[row0, col1],
                self.heights[row1, col0],
                self.heights[row1, col1],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            return None
        row_t = float(row) - row0
        col_t = float(col) - col0
        top = values[0] * (1.0 - col_t) + values[1] * col_t
        bottom = values[2] * (1.0 - col_t) + values[3] * col_t
        return float(top * (1.0 - row_t) + bottom * row_t)


@dataclass(frozen=True)
class TerrainCostMap:
    cost: np.ndarray
    slope_degrees: np.ndarray
    roughness: np.ndarray
    clearance: np.ndarray
    blocked: np.ndarray
    config: TerrainCostConfig

    def __post_init__(self) -> None:
        shape = self.cost.shape
        arrays = (self.slope_degrees, self.roughness, self.clearance, self.blocked)
        if self.cost.ndim != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("All terrain cost arrays must be 2D and share one shape")

    def is_walkable(self, point: GridPoint, *, min_clearance: float = 0.0) -> bool:
        row, col = point
        if not _in_bounds(self.cost.shape, row, col):
            return False
        return bool(
            not self.blocked[row, col]
            and np.isfinite(self.cost[row, col])
            and self.clearance[row, col] >= min_clearance
        )


@dataclass(frozen=True)
class TerrainPath:
    points: list[GridPoint]
    total_cost: float


def load_adt_v9(path: str | Path) -> np.ndarray:
    source = Path(path)
    data = source.read_bytes()
    heights = np.full((V9_SIZE, V9_SIZE), np.nan, dtype=np.float32)
    assigned = np.zeros((V9_SIZE, V9_SIZE), dtype=bool)
    chunk_indexes: set[tuple[int, int]] = set()
    search_offset = 0

    while True:
        chunk_offset = data.find(b"KNCM", search_offset)
        if chunk_offset < 0:
            break
        if chunk_offset + 136 > len(data):
            raise ValueError(f"Truncated MCNK header in {source} at {chunk_offset}")

        chunk_size = struct.unpack_from("<I", data, chunk_offset + 4)[0]
        chunk_end = chunk_offset + 8 + chunk_size
        if chunk_end > len(data):
            raise ValueError(f"MCNK at {chunk_offset} exceeds {source} size")

        ix, iy = struct.unpack_from("<II", data, chunk_offset + 12)
        if not 0 <= ix < 16 or not 0 <= iy < 16:
            raise ValueError(f"Invalid MCNK index ({ix}, {iy}) in {source}")
        if (ix, iy) in chunk_indexes:
            raise ValueError(f"Duplicate MCNK index ({ix}, {iy}) in {source}")

        mcvt_relative = struct.unpack_from("<I", data, chunk_offset + 28)[0]
        base_height = struct.unpack_from("<f", data, chunk_offset + 120)[0]
        mcvt_offset = chunk_offset + mcvt_relative
        if mcvt_offset < chunk_offset + 8 or mcvt_offset + 8 > chunk_end:
            raise ValueError(f"Invalid MCVT offset for MCNK ({ix}, {iy}) in {source}")
        if data[mcvt_offset : mcvt_offset + 4] != b"TVCM":
            raise ValueError(f"Missing TVCM for MCNK ({ix}, {iy}) in {source}")

        mcvt_size = struct.unpack_from("<I", data, mcvt_offset + 4)[0]
        if mcvt_size < 145 * 4 or mcvt_offset + 8 + mcvt_size > chunk_end:
            raise ValueError(f"Invalid MCVT size {mcvt_size} for MCNK ({ix}, {iy}) in {source}")
        relative = np.asarray(struct.unpack_from("<145f", data, mcvt_offset + 8), dtype=np.float32)

        for local_row in range(9):
            row = iy * 8 + local_row
            col_start = ix * 8
            values = base_height + relative[local_row * 17 : local_row * 17 + 9]
            existing_mask = assigned[row, col_start : col_start + 9]
            if np.any(existing_mask):
                existing = heights[row, col_start : col_start + 9][existing_mask]
                incoming = values[existing_mask]
                if np.max(np.abs(existing - incoming), initial=0.0) > 1.0:
                    raise ValueError(f"MCNK shared-vertex mismatch at ({ix}, {iy}) in {source}")
            heights[row, col_start : col_start + 9] = values
            assigned[row, col_start : col_start + 9] = True

        chunk_indexes.add((ix, iy))
        search_offset = chunk_end

    if len(chunk_indexes) != 256 or not np.all(assigned):
        raise ValueError(
            f"Expected complete 16x16 MCNK terrain in {source}; "
            f"found {len(chunk_indexes)} chunks and {int(assigned.sum())}/{assigned.size} vertices"
        )
    return heights


def parse_adt_tile_xy(path: str | Path) -> tuple[int, int]:
    match = re.search(r"_(-?\d+)_(-?\d+)\.adt$", Path(path).name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse ADT tile coordinates from {Path(path).name}")
    return int(match.group(1)), int(match.group(2))


def build_terrain_raster(
    adt_paths: Iterable[str | Path],
    *,
    boundary_tolerance: float = 1.0,
) -> TerrainRaster:
    paths = sorted((Path(path) for path in adt_paths), key=lambda path: parse_adt_tile_xy(path))
    if not paths:
        raise ValueError("No ADT paths supplied")
    if boundary_tolerance < 0.0:
        raise ValueError("Boundary tolerance must be non-negative")

    tiles: dict[tuple[int, int], np.ndarray] = {}
    for path in paths:
        tile = parse_adt_tile_xy(path)
        if tile in tiles:
            raise ValueError(f"Duplicate ADT tile {tile}")
        tiles[tile] = load_adt_v9(path)

    tile_min_x = min(tile[0] for tile in tiles)
    tile_max_x = max(tile[0] for tile in tiles)
    tile_min_y = min(tile[1] for tile in tiles)
    tile_max_y = max(tile[1] for tile in tiles)
    shape = (
        (tile_max_y - tile_min_y + 1) * TILE_GRID_CELLS + 1,
        (tile_max_x - tile_min_x + 1) * TILE_GRID_CELLS + 1,
    )
    mosaic = np.full(shape, np.nan, dtype=np.float32)

    for (tile_x, tile_y), tile_heights in sorted(tiles.items()):
        row = (tile_y - tile_min_y) * TILE_GRID_CELLS
        col = (tile_x - tile_min_x) * TILE_GRID_CELLS
        target = mosaic[row : row + V9_SIZE, col : col + V9_SIZE]
        overlap = np.isfinite(target)
        if np.any(overlap):
            mismatch = np.abs(target[overlap] - tile_heights[overlap])
            if float(np.max(mismatch, initial=0.0)) > boundary_tolerance:
                raise ValueError(f"ADT shared boundary mismatch at tile ({tile_x}, {tile_y})")
        write_mask = np.isfinite(tile_heights)
        target[write_mask] = tile_heights[write_mask]

    return TerrainRaster(
        heights=mosaic,
        tile_min_x=tile_min_x,
        tile_min_y=tile_min_y,
        tile_max_x=tile_max_x,
        tile_max_y=tile_max_y,
    )


def build_terrain_cost_map(
    raster: TerrainRaster,
    config: TerrainCostConfig | None = None,
) -> TerrainCostMap:
    cost_config = config or TerrainCostConfig()
    heights = np.asarray(raster.heights, dtype=np.float32)
    valid = np.isfinite(heights)
    gradient_row = _finite_gradient(heights, valid, axis=0, spacing=GRID_SPACING)
    gradient_col = _finite_gradient(heights, valid, axis=1, spacing=GRID_SPACING)
    slope = np.degrees(np.arctan(np.hypot(gradient_row, gradient_col))).astype(np.float32)

    kernel = np.ones((3, 3), dtype=np.uint8)
    max_input = np.where(valid, heights, np.float32(-1.0e20))
    min_input = np.where(valid, heights, np.float32(1.0e20))
    local_max = cv2.dilate(max_input, kernel)
    local_min = cv2.erode(min_input, kernel)
    roughness = (local_max - local_min).astype(np.float32)
    roughness[~valid] = np.nan

    blocked = ~valid | ~np.isfinite(slope) | (slope >= cost_config.hard_slope_degrees)
    walkable_u8 = (~blocked).astype(np.uint8)
    clearance = cv2.distanceTransform(walkable_u8, cv2.DIST_L2, 3).astype(np.float32)

    slope_span = cost_config.hard_slope_degrees - cost_config.soft_slope_degrees
    slope_fraction = np.clip((slope - cost_config.soft_slope_degrees) / slope_span, 0.0, 1.0)
    slope_penalty = cost_config.slope_weight * np.square(slope_fraction)
    roughness_penalty = cost_config.roughness_weight * np.nan_to_num(roughness, nan=0.0)
    if cost_config.clearance_cells > 0.0:
        clearance_fraction = np.clip(
            (cost_config.clearance_cells - clearance) / cost_config.clearance_cells,
            0.0,
            1.0,
        )
        clearance_penalty = cost_config.clearance_weight * np.square(clearance_fraction)
    else:
        clearance_penalty = np.zeros_like(clearance)

    cost = (1.0 + slope_penalty + roughness_penalty + clearance_penalty).astype(np.float32)
    cost[blocked] = np.inf
    return TerrainCostMap(
        cost=cost,
        slope_degrees=slope,
        roughness=roughness,
        clearance=clearance,
        blocked=blocked,
        config=cost_config,
    )


def nearest_walkable(
    cost_map: TerrainCostMap,
    point: GridPoint,
    max_radius: float,
    *,
    min_clearance: float = 0.0,
) -> GridPoint | None:
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")
    start_row, start_col = point
    radius = int(math.ceil(max_radius))
    candidates: list[tuple[float, int, int]] = []
    for row in range(start_row - radius, start_row + radius + 1):
        for col in range(start_col - radius, start_col + radius + 1):
            distance = math.hypot(row - start_row, col - start_col)
            if distance <= max_radius and cost_map.is_walkable((row, col), min_clearance=min_clearance):
                candidates.append((distance, row, col))
    if not candidates:
        return None
    _, row, col = min(candidates)
    return row, col


def astar(
    cost_map: TerrainCostMap,
    start: GridPoint,
    goal: GridPoint,
    *,
    search_bounds: tuple[int, int, int, int] | None = None,
    max_expansions: int | None = None,
) -> TerrainPath | None:
    if not cost_map.is_walkable(start) or not cost_map.is_walkable(goal):
        return None
    if max_expansions is not None and max_expansions <= 0:
        raise ValueError("max_expansions must be positive")

    rows, cols = cost_map.cost.shape
    if search_bounds is None:
        row_min, row_max, col_min, col_max = 0, rows - 1, 0, cols - 1
    else:
        row_min, row_max, col_min, col_max = search_bounds
        row_min = max(0, int(row_min))
        row_max = min(rows - 1, int(row_max))
        col_min = max(0, int(col_min))
        col_max = min(cols - 1, int(col_max))
        if row_min > row_max or col_min > col_max:
            raise ValueError("Invalid A* search bounds")
    if not _inside_bounds(start, row_min, row_max, col_min, col_max):
        return None
    if not _inside_bounds(goal, row_min, row_max, col_min, col_max):
        return None

    finite_costs = cost_map.cost[
        row_min : row_max + 1,
        col_min : col_max + 1,
    ]
    finite_costs = finite_costs[np.isfinite(finite_costs)]
    if finite_costs.size == 0:
        return None
    minimum_cost = float(np.min(finite_costs))

    sequence = count()
    frontier: list[tuple[float, float, int, GridPoint]] = []
    heapq.heappush(frontier, (_heuristic(start, goal, minimum_cost), 0.0, next(sequence), start))
    came_from: dict[GridPoint, GridPoint] = {}
    best_cost: dict[GridPoint, float] = {start: 0.0}
    expansions = 0

    while frontier:
        _, current_cost, _, current = heapq.heappop(frontier)
        if current_cost > best_cost.get(current, math.inf) + 1.0e-9:
            continue
        if current == goal:
            return TerrainPath(points=_reconstruct_path(came_from, goal), total_cost=current_cost)
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            return None

        row, col = current
        for delta_row, delta_col, step_length in _NEIGHBORS:
            next_point = row + delta_row, col + delta_col
            next_row, next_col = next_point
            if not _inside_bounds(next_point, row_min, row_max, col_min, col_max):
                continue
            if not cost_map.is_walkable(next_point):
                continue
            if delta_row != 0 and delta_col != 0:
                if not cost_map.is_walkable((row + delta_row, col)):
                    continue
                if not cost_map.is_walkable((row, col + delta_col)):
                    continue

            transition = step_length * (
                float(cost_map.cost[row, col]) + float(cost_map.cost[next_row, next_col])
            ) * 0.5
            candidate_cost = current_cost + transition
            if candidate_cost + 1.0e-9 >= best_cost.get(next_point, math.inf):
                continue
            best_cost[next_point] = candidate_cost
            came_from[next_point] = current
            priority = candidate_cost + _heuristic(next_point, goal, minimum_cost)
            heapq.heappush(frontier, (priority, candidate_cost, next(sequence), next_point))
    return None


def supercover_line(start: GridPoint, end: GridPoint) -> list[GridPoint]:
    row0, col0 = start
    row1, col1 = end
    delta_col = col1 - col0
    delta_row = row1 - row0
    count_col = abs(delta_col)
    count_row = abs(delta_row)
    sign_col = 1 if delta_col > 0 else -1 if delta_col < 0 else 0
    sign_row = 1 if delta_row > 0 else -1 if delta_row < 0 else 0
    col = col0
    row = row0
    index_col = 0
    index_row = 0
    points: list[GridPoint] = [(row, col)]

    while index_col < count_col or index_row < count_row:
        decision = (1 + 2 * index_col) * count_row - (1 + 2 * index_row) * count_col
        if decision == 0:
            if sign_col:
                _append_unique(points, (row, col + sign_col))
            if sign_row:
                _append_unique(points, (row + sign_row, col))
            col += sign_col
            row += sign_row
            index_col += 1
            index_row += 1
        elif decision < 0:
            col += sign_col
            index_col += 1
        else:
            row += sign_row
            index_row += 1
        _append_unique(points, (row, col))
    return points


def line_is_walkable(
    cost_map: TerrainCostMap,
    start: GridPoint,
    end: GridPoint,
    *,
    min_clearance: float = 0.0,
) -> bool:
    return all(
        cost_map.is_walkable(point, min_clearance=min_clearance)
        for point in supercover_line(start, end)
    )


def simplify_path(
    cost_map: TerrainCostMap,
    points: Sequence[GridPoint],
    max_spacing_cells: float,
    *,
    min_clearance: float = 0.0,
) -> list[GridPoint]:
    if max_spacing_cells <= 0.0:
        raise ValueError("max_spacing_cells must be positive")
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]

    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        best = anchor + 1
        candidate = anchor + 1
        while candidate < len(points):
            if math.dist(points[anchor], points[candidate]) > max_spacing_cells:
                break
            if line_is_walkable(
                cost_map,
                points[anchor],
                points[candidate],
                min_clearance=min_clearance,
            ):
                best = candidate
            candidate += 1
        simplified.append(points[best])
        anchor = best
    return simplified


def path_length(points: Sequence[GridPoint], *, spacing: float = GRID_SPACING) -> float:
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return float(
        sum(math.dist(start, end) * spacing for start, end in zip(points, points[1:]))
    )


def _finite_gradient(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    axis: int,
    spacing: float,
) -> np.ndarray:
    gradient = np.full(values.shape, np.nan, dtype=np.float32)
    if axis == 0:
        center = values[1:-1, :]
        before = values[:-2, :]
        after = values[2:, :]
        center_valid = valid[1:-1, :]
        before_valid = valid[:-2, :]
        after_valid = valid[2:, :]
        both = center_valid & before_valid & after_valid
        forward = center_valid & ~before_valid & after_valid
        backward = center_valid & before_valid & ~after_valid
        interior = gradient[1:-1, :]
    elif axis == 1:
        center = values[:, 1:-1]
        before = values[:, :-2]
        after = values[:, 2:]
        center_valid = valid[:, 1:-1]
        before_valid = valid[:, :-2]
        after_valid = valid[:, 2:]
        both = center_valid & before_valid & after_valid
        forward = center_valid & ~before_valid & after_valid
        backward = center_valid & before_valid & ~after_valid
        interior = gradient[:, 1:-1]
    else:
        raise ValueError("axis must be 0 or 1")

    interior[both] = (after[both] - before[both]) / (2.0 * spacing)
    interior[forward] = (after[forward] - center[forward]) / spacing
    interior[backward] = (center[backward] - before[backward]) / spacing

    if axis == 0:
        top = valid[0, :] & valid[1, :]
        bottom = valid[-1, :] & valid[-2, :]
        gradient[0, top] = (values[1, top] - values[0, top]) / spacing
        gradient[-1, bottom] = (values[-1, bottom] - values[-2, bottom]) / spacing
    else:
        left = valid[:, 0] & valid[:, 1]
        right = valid[:, -1] & valid[:, -2]
        gradient[left, 0] = (values[left, 1] - values[left, 0]) / spacing
        gradient[right, -1] = (values[right, -1] - values[right, -2]) / spacing
    return gradient


_NEIGHBORS = (
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _heuristic(point: GridPoint, goal: GridPoint, minimum_cost: float) -> float:
    return math.dist(point, goal) * minimum_cost


def _reconstruct_path(came_from: dict[GridPoint, GridPoint], goal: GridPoint) -> list[GridPoint]:
    points = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        points.append(current)
    points.reverse()
    return points


def _inside_bounds(
    point: GridPoint,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> bool:
    row, col = point
    return row_min <= row <= row_max and col_min <= col <= col_max


def _in_bounds(shape: tuple[int, ...], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _append_unique(points: list[GridPoint], point: GridPoint) -> None:
    if not points or points[-1] != point:
        points.append(point)
