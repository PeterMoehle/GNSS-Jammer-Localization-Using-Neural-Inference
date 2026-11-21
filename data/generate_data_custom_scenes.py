"""
Totally vibecoded version of generate_data
Incorporates fixes:

- Use our scenes rather than the predefined one
- Use osm tags to find the walkable area, by subtracting buildings/roofs and water from the walkable area, that is determined via roas/paths/areas...
- Sionna: Loading scenes with merge_shapes = True
- for testing we should have two seperate subsets, one for in-distribution (some of the trained scenes, but unseen episodes) and one for out-of-distribution (some held out scenes)
- Jammer Route trajectories along walkable streets/paths
- Jammer speed should maybe be increased to 5m/s
- Use multiple sensor placements within the training set
- Subregions sampling within a scene, but dont use the buffer zone. The buffer is just for raytracing, not actually for interest
- Because antennas are isotropic (pattern="iso"), swap TX and RX in PathSolver: place the $N$ stationary sensors as TX and all $T$ trajectory points (plus extra candidate points) as RX in a single PathSolver call per episode. This cuts ray-tracing calls by ~4–8× and lets you reuse one RT pass across multiple sampled sensor layouts.

Improvement for later:
- Data should be stored as one big file, including the dataconfig
"""

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Literal, get_args

import drjit as dr
import h5py
import mitsuba as mi
import numpy as np
import sionna.rt as rt
from scipy.ndimage import binary_erosion, label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from tqdm import tqdm

PlacementType = Literal["random", "spread", "lhs"]


@dataclass(frozen=True)
class DataConfig:
    scene_name: str = "würzburg"
    baked_scenes_dir: Path = Path("workspace_inputs/baked_scenes")
    frequency_hz: float = 1.57542e9
    tx_power_dbm: float = 20.0
    sensitivity_dbm: float = -110.0
    rssi_noise_std_db: float = 0.0
    max_depth: int = 3
    diffraction: bool = True
    samples_per_src: int = 20_000
    max_num_paths_per_src: int = 250_000
    batch_size: int = 8  # Kept for CLI/JSON config backwards compatibility
    roi_center_xy_m: tuple[float, float] = (0.0, 0.0)
    roi_half_size_m: float = 500.0
    grid_resolution_m: float = 2.0
    ground_tolerance_m: float = 0.5
    erosion_iterations: int = 1
    include_parks_as_walkable: bool = True
    prefer_streets_for_jammer: bool = True
    use_precomputed_walkable_map: bool = True
    cache_walkable_map_if_missing: bool = True
    n_sensors: int = 20
    sensor_height_m: float = 1.5
    jammer_height_m: float = 1.5
    sensor_placements: tuple[PlacementType, ...] = ("spread", "random", "lhs")
    graph_neighbors: int = 3
    episodes: int = 32
    n_steps: int = 60
    dt_s: float = 1.0
    speed_mps: float = 5.0
    heading_std_rad: float = 0.25
    seed: int = 42
    test_fraction: float = 0.2
    output_dir: Path = Path("data/generated")
    output_name: str = ""  # Empty: <scene>_seed<seed>.h5
    overwrite: bool = False
    debug: bool = False
    chunk_episodes: int = 64
    roi_mode: Literal["manual", "auto", "random"] = "random"


@dataclass
class WalkableMap:
    x_m: np.ndarray
    y_m: np.ndarray
    height_m: np.ndarray
    free: np.ndarray
    street: np.ndarray
    resolution_m: float

    def lookup(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(xy)
        ix = np.floor((xy[:, 0] - self.x_m[0]) / self.resolution_m + 0.5).astype(int)
        iy = np.floor((xy[:, 1] - self.y_m[0]) / self.resolution_m + 0.5).astype(int)
        inside = (0 <= ix) & (ix < self.free.shape[1]) & (0 <= iy) & (iy < self.free.shape[0])
        free = np.zeros(len(xy), dtype=bool)
        height = np.full(len(xy), np.nan, dtype=np.float32)
        free[inside] = self.free[iy[inside], ix[inside]]
        height[inside] = self.height_m[iy[inside], ix[inside]]
        return free, height

    def lookup_street(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        ix = np.floor((xy[:, 0] - self.x_m[0]) / self.resolution_m + 0.5).astype(int)
        iy = np.floor((xy[:, 1] - self.y_m[0]) / self.resolution_m + 0.5).astype(int)
        inside = (0 <= ix) & (ix < self.street.shape[1]) & (0 <= iy) & (iy < self.street.shape[0])
        on_street = np.zeros(len(xy), dtype=bool)
        on_street[inside] = self.street[iy[inside], ix[inside]]
        return on_street

    def candidates(self, streets_only: bool = False) -> np.ndarray:
        mask = self.street if (streets_only and np.any(self.street)) else self.free
        iy, ix = np.nonzero(mask)
        return np.column_stack((self.x_m[ix], self.y_m[iy]))


@dataclass
class PrecomputedSceneMap:
    x_m: np.ndarray
    y_m: np.ndarray
    height_m: np.ndarray
    street: np.ndarray
    park: np.ndarray
    resolution_m: float

    def extract_roi(
        self,
        center: tuple[float, float],
        roi_half_size_m: float,
        include_parks: bool = True,
        erosion_iterations: int = 1,
        min_walkable_cells: int = 1,
    ) -> WalkableMap | None:
        """Fast array slice of a localized ROI from the precomputed full-scene grid."""
        n = int(np.floor(2.0 * roi_half_size_m / self.resolution_m))
        x0 = center[0] - roi_half_size_m
        y0 = center[1] - roi_half_size_m

        ix0 = int(np.floor((x0 - self.x_m[0]) / self.resolution_m + 0.5))
        iy0 = int(np.floor((y0 - self.y_m[0]) / self.resolution_m + 0.5))
        ix1 = ix0 + n
        iy1 = iy0 + n

        if ix0 < 0 or iy0 < 0 or ix1 > len(self.x_m) or iy1 > len(self.y_m):
            return None

        sub_street = self.street[iy0:iy1, ix0:ix1].copy()
        sub_walkable = sub_street | self.park[iy0:iy1, ix0:ix1] if include_parks else sub_street.copy()

        if erosion_iterations:
            sub_walkable = binary_erosion(sub_walkable, iterations=erosion_iterations)
            sub_street = sub_street & sub_walkable

        components, count = label(sub_walkable)
        if not count:
            return None

        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if sizes.max() < min_walkable_cells:
            return None

        main_free = components == sizes.argmax()
        main_street = sub_street & main_free
        return WalkableMap(
            x_m=self.x_m[ix0:ix1].copy(),
            y_m=self.y_m[iy0:iy1].copy(),
            height_m=self.height_m[iy0:iy1, ix0:ix1].copy(),
            free=main_free,
            street=main_street,
            resolution_m=self.resolution_m,
        )


def resolve_baked_scene(cfg: DataConfig) -> tuple[str | Path, dict, Path | None]:
    """Resolve baked Mitsuba XML path, metadata, and scene directory."""
    scene_dir = cfg.baked_scenes_dir / cfg.scene_name
    meta_file = scene_dir / "scene_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        xml_path = scene_dir / meta["xml_filename"]
        if not xml_path.exists():
            raise FileNotFoundError(f"Baked XML missing at {xml_path}")
        return xml_path, meta, scene_dir

    direct_xml = Path(cfg.scene_name)
    if direct_xml.suffix == ".xml" and direct_xml.exists():
        meta_path = direct_xml.parent / "scene_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return direct_xml, meta, direct_xml.parent

    if hasattr(rt.scene, cfg.scene_name):
        return getattr(rt.scene, cfg.scene_name), {}, None

    raise FileNotFoundError(
        f"Scene '{cfg.scene_name}' not found in {cfg.baked_scenes_dir} and is not a built-in Sionna scene."
    )


def _classify_scene_shapes(scene: rt.Scene, include_parks: bool = True) -> tuple[list, list, list]:
    """Classify merged Mitsuba shapes using BSDF material IDs / shape IDs."""
    street_shapes = []
    park_shapes = []
    obstacle_shapes = []
    ground_shapes = []

    street_tokens = ("road", "path", "pedestrian", "steps", "cycleway", "footway")
    obstacle_tokens = ("building", "roof", "wall", "water")

    for shape in scene.mi_scene.shapes():
        shape_id = (shape.id() or "").lower()
        bsdf = shape.bsdf()
        bsdf_id = (bsdf.id() if bsdf is not None and bsdf.id() else "").lower()
        tag = f"{shape_id} {bsdf_id}"

        if any(tok in tag for tok in obstacle_tokens):
            obstacle_shapes.append(shape)
        elif any(tok in tag for tok in street_tokens):
            street_shapes.append(shape)
        elif "park" in tag:
            if include_parks:
                park_shapes.append(shape)
        elif "ground" in tag or "plane" in tag:
            ground_shapes.append(shape)

    if not street_shapes and not park_shapes and ground_shapes:
        street_shapes = ground_shapes

    if not street_shapes and not park_shapes:
        raise RuntimeError(
            "Could not identify any walkable shapes (roads/paths/parks/ground) in the loaded Mitsuba scene."
        )
    return street_shapes, park_shapes, obstacle_shapes


def build_and_cache_precomputed_map(
    scene: rt.Scene,
    scene_dir: Path,
    usable_half_range_m: float,
    resolution_m: float = 2.0,
    ray_batch_size: int = 1_000_000,
) -> PrecomputedSceneMap:
    """Build and store walkable_map.npz inside scene_dir if not yet present."""
    bbox = scene.mi_scene.bbox()
    z_top = float(bbox.max.z) + 10.0
    n = int(np.floor(2.0 * usable_half_range_m / resolution_m))
    coords = -usable_half_range_m + (np.arange(n, dtype=np.float32) + 0.5) * resolution_m
    xx, yy = np.meshgrid(coords, coords)
    total_pts = xx.size

    street_shapes, park_shapes, obstacle_shapes = _classify_scene_shapes(scene, include_parks=True)
    street_ptrs = [mi.ShapePtr(s) for s in street_shapes]
    park_ptrs = [mi.ShapePtr(s) for s in park_shapes]
    obstacle_ptrs = [mi.ShapePtr(s) for s in obstacle_shapes]

    height_flat = np.full(total_pts, np.nan, dtype=np.float32)
    street_flat = np.zeros(total_pts, dtype=bool)
    park_flat = np.zeros(total_pts, dtype=bool)
    x_flat, y_flat = xx.ravel(), yy.ravel()

    for start in range(0, total_pts, ray_batch_size):
        end = min(start + ray_batch_size, total_pts)
        chunk_len = end - start
        origins = mi.Point3f(
            x_flat[start:end],
            y_flat[start:end],
            np.full(chunk_len, z_top, dtype=np.float32),
        )
        hits = scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0, 0, -1)))
        valid = np.asarray(hits.is_valid().numpy())
        height_flat[start:end] = np.where(valid, z_top - np.asarray(hits.t.numpy()), np.nan)

        is_street = np.zeros(chunk_len, dtype=bool)
        for ptr in street_ptrs:
            is_street |= np.asarray((hits.shape == ptr).numpy())

        is_park = np.zeros(chunk_len, dtype=bool)
        for ptr in park_ptrs:
            is_park |= np.asarray((hits.shape == ptr).numpy())

        is_obstacle = np.zeros(chunk_len, dtype=bool)
        for ptr in obstacle_ptrs:
            is_obstacle |= np.asarray((hits.shape == ptr).numpy())

        street_flat[start:end] = valid & is_street & (~is_obstacle)
        park_flat[start:end] = valid & is_park & (~is_obstacle)

    out_file = scene_dir / "walkable_map.npz"
    height_grid = height_flat.reshape(n, n)
    street_grid = street_flat.reshape(n, n)
    park_grid = park_flat.reshape(n, n)

    np.savez_compressed(
        out_file,
        x_m=coords,
        y_m=coords,
        height_m=height_grid,
        street=street_grid,
        park=park_grid,
        resolution_m=np.float32(resolution_m),
    )
    print(f"Cached precomputed walkable map -> {out_file}")
    return PrecomputedSceneMap(
        x_m=coords,
        y_m=coords,
        height_m=height_grid,
        street=street_grid,
        park=park_grid,
        resolution_m=resolution_m,
    )


def load_scene_and_walkable_map(cfg: DataConfig) -> tuple[rt.Scene, dict, PrecomputedSceneMap | None]:
    """Load baked scene with merge_shapes=True and load/build its cached walkable map."""
    scene_source, meta, scene_dir = resolve_baked_scene(cfg)
    scene = load_scene(str(scene_source), merge_shapes=True)
    scene.frequency = cfg.frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    precomputed: PrecomputedSceneMap | None = None
    if cfg.use_precomputed_walkable_map and scene_dir is not None:
        walk_file = scene_dir / meta.get("walkable_map_filename", "walkable_map.npz")
        if walk_file.exists():
            with np.load(walk_file) as data:
                stored_res = float(data["resolution_m"])
                if np.isclose(stored_res, cfg.grid_resolution_m, atol=1e-4):
                    precomputed = PrecomputedSceneMap(
                        x_m=data["x_m"],
                        y_m=data["y_m"],
                        height_m=data["height_m"],
                        street=data["street"],
                        park=data["park"],
                        resolution_m=stored_res,
                    )
        elif cfg.cache_walkable_map_if_missing and "usable_xy_half_range_m" in meta:
            precomputed = build_and_cache_precomputed_map(
                scene=scene,
                scene_dir=scene_dir,
                usable_half_range_m=float(meta["usable_xy_half_range_m"]),
                resolution_m=cfg.grid_resolution_m,
            )

    return scene, meta, precomputed


def sample_episode_grid(
    scene: rt.Scene,
    cfg: DataConfig,
    rng: np.random.Generator,
    scene_meta: dict | None = None,
    precomputed_map: PrecomputedSceneMap | None = None,
    max_retries: int = 100,
) -> WalkableMap:
    """
    Sample a localized ROI strictly inside the non-buffer usable bounds.
    Uses fast slicing from `precomputed_map` when available, or falls back to ray-casting.
    """
    bbox = scene.mi_scene.bbox()
    if scene_meta and "usable_xy_half_range_m" in scene_meta:
        half_usable = float(scene_meta["usable_xy_half_range_m"])
        min_x, max_x = -half_usable, half_usable
        min_y, max_y = -half_usable, half_usable
    else:
        min_x, max_x = float(bbox.min.x), float(bbox.max.x)
        min_y, max_y = float(bbox.min.y), float(bbox.max.y)

    safe_margin = cfg.roi_half_size_m
    if (max_x - min_x) < 2 * safe_margin or (max_y - min_y) < 2 * safe_margin:
        raise ValueError(
            f"ROI size ({2 * safe_margin:g} m) exceeds usable non-buffer scene bounds "
            f"({max_x - min_x:g} m x {max_y - min_y:g} m) for '{cfg.scene_name}'."
        )

    n = int(np.floor(2 * cfg.roi_half_size_m / cfg.grid_resolution_m))
    if n < 2:
        raise ValueError("ROI too small for grid resolution")

    # Fast path: slice directly from precomputed walkable_map.npz
    if precomputed_map is not None:
        for _ in range(max_retries):
            if cfg.roi_mode == "random":
                center = (
                    rng.uniform(min_x + safe_margin, max_x - safe_margin),
                    rng.uniform(min_y + safe_margin, max_y - safe_margin),
                )
            elif cfg.roi_mode == "auto":
                center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            else:
                center = cfg.roi_center_xy_m
                if not (
                    min_x + safe_margin <= center[0] <= max_x - safe_margin
                    and min_y + safe_margin <= center[1] <= max_y - safe_margin
                ):
                    raise ValueError(
                        f"Manual roi_center_xy_m={center} with roi_half_size_m={safe_margin} "
                        f"extends into the ray-tracing buffer zone [{min_x}, {max_x}] x [{min_y}, {max_y}]."
                    )

            grid = precomputed_map.extract_roi(
                center=center,
                roi_half_size_m=cfg.roi_half_size_m,
                include_parks=cfg.include_parks_as_walkable,
                erosion_iterations=cfg.erosion_iterations,
                min_walkable_cells=cfg.n_sensors,
            )
            if grid is not None:
                return grid

        raise ValueError(
            f"Could not find a valid walkable area in precomputed map with >= {cfg.n_sensors} cells after {max_retries} attempts."
        )

    # Fallback path: on-the-fly ray intersection against Mitsuba shapes
    z_top = float(bbox.max.z) + 10.0
    street_shapes, park_shapes, obstacle_shapes = _classify_scene_shapes(
        scene, include_parks=cfg.include_parks_as_walkable
    )
    street_ptrs = [mi.ShapePtr(s) for s in street_shapes]
    park_ptrs = [mi.ShapePtr(s) for s in park_shapes]
    obstacle_ptrs = [mi.ShapePtr(s) for s in obstacle_shapes]

    for _ in range(max_retries):
        if cfg.roi_mode == "random":
            center = (
                rng.uniform(min_x + safe_margin, max_x - safe_margin),
                rng.uniform(min_y + safe_margin, max_y - safe_margin),
            )
        elif cfg.roi_mode == "auto":
            center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        else:
            center = cfg.roi_center_xy_m
            if not (
                min_x + safe_margin <= center[0] <= max_x - safe_margin
                and min_y + safe_margin <= center[1] <= max_y - safe_margin
            ):
                raise ValueError(
                    f"Manual roi_center_xy_m={center} with roi_half_size_m={safe_margin} "
                    f"extends into the ray-tracing buffer zone [{min_x}, {max_x}] x [{min_y}, {max_y}]."
                )

        offsets = -cfg.roi_half_size_m + (np.arange(n) + 0.5) * cfg.grid_resolution_m
        x_m, y_m = offsets + center[0], offsets + center[1]
        xx, yy = np.meshgrid(x_m, y_m)

        origins = mi.Point3f(
            xx.ravel().astype(np.float32),
            yy.ravel().astype(np.float32),
            np.full(xx.size, z_top, dtype=np.float32),
        )
        hits = scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0, 0, -1)))
        valid = np.asarray(hits.is_valid().numpy()).reshape(xx.shape)
        height = np.where(valid, z_top - np.asarray(hits.t.numpy()).reshape(xx.shape), np.nan)

        is_street_surface = np.zeros(xx.shape, dtype=bool)
        for s_ptr in street_ptrs:
            is_street_surface |= np.asarray((hits.shape == s_ptr).numpy()).reshape(xx.shape)

        is_park_surface = np.zeros(xx.shape, dtype=bool)
        for p_ptr in park_ptrs:
            is_park_surface |= np.asarray((hits.shape == p_ptr).numpy()).reshape(xx.shape)

        is_obstacle = np.zeros(xx.shape, dtype=bool)
        for o_ptr in obstacle_ptrs:
            is_obstacle |= np.asarray((hits.shape == o_ptr).numpy()).reshape(xx.shape)

        raw_street = valid & is_street_surface & (~is_obstacle)
        raw_walkable = valid & (is_street_surface | is_park_surface) & (~is_obstacle)

        if cfg.erosion_iterations:
            raw_walkable = binary_erosion(raw_walkable, iterations=cfg.erosion_iterations)
            raw_street = raw_street & raw_walkable

        components, count = label(raw_walkable)
        if not count:
            continue

        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if sizes.max() >= cfg.n_sensors:
            main_free = components == sizes.argmax()
            main_street = raw_street & main_free
            return WalkableMap(
                x_m=x_m,
                y_m=y_m,
                height_m=height,
                free=main_free,
                street=main_street,
                resolution_m=cfg.grid_resolution_m,
            )

    raise ValueError(
        f"Could not find a valid OSM walkable area with >= {cfg.n_sensors} cells after {max_retries} attempts."
    )


def generate_trajectories(grids: list[WalkableMap], cfg: DataConfig, rng: np.random.Generator):
    """Generate street/path-following jammer trajectories."""
    positions = np.empty((cfg.episodes, cfg.n_steps, 3), dtype=np.float32)
    step = cfg.speed_mps * cfg.dt_s
    checks = max(3, int(np.ceil(step / (cfg.grid_resolution_m / 2))) + 1)

    turn_offsets = np.concatenate([
        np.linspace(-0.25, 0.25, 9),
        np.linspace(-np.pi / 2, np.pi / 2, 13),
        np.linspace(-np.pi, np.pi, 16),
    ])

    for e in range(cfg.episodes):
        grid = grids[e]
        use_streets = cfg.prefer_streets_for_jammer and np.any(grid.street)
        candidates = grid.candidates(streets_only=use_streets)
        xy = candidates[rng.integers(len(candidates))].copy()
        heading = rng.uniform(0, 2 * np.pi)

        for t in range(cfg.n_steps):
            _, z = grid.lookup(xy)
            positions[e, t] = (*xy, z[0] + cfg.jammer_height_m)
            if t == cfg.n_steps - 1 or step == 0:
                continue

            moved = False
            best_fallback = None

            jitter = rng.normal(0, cfg.heading_std_rad)
            for idx, delta_angle in enumerate(turn_offsets):
                angle = heading + jitter + delta_angle if idx < 9 else heading + delta_angle
                move = step * np.array([np.cos(angle), np.sin(angle)])
                samples = xy + np.linspace(0, 1, checks)[:, None] * move

                if not grid.lookup(samples)[0].all():
                    continue

                if use_streets:
                    if grid.lookup_street(samples).all():
                        xy, heading = xy + move, angle
                        moved = True
                        break
                    elif best_fallback is None:
                        best_fallback = (xy + move, angle)
                else:
                    xy, heading = xy + move, angle
                    moved = True
                    break

            if not moved and best_fallback is not None:
                xy, heading = best_fallback

    velocity = np.zeros_like(positions)
    velocity[:, 1:] = np.diff(positions, axis=1) / cfg.dt_s
    return positions, velocity


def sample_sensor_layout(
    grid: WalkableMap,
    cfg: DataConfig,
    rng: np.random.Generator,
    placement_mode: PlacementType,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = grid.candidates(streets_only=False)
    if not 1 <= cfg.n_sensors <= len(candidates):
        raise ValueError("Invalid n_sensors")

    if placement_mode == "random":
        chosen = rng.choice(len(candidates), cfg.n_sensors, replace=False)
    elif placement_mode == "lhs":
        sampler = qmc.LatinHypercube(d=2, seed=rng)
        unit_samples = sampler.random(n=cfg.n_sensors)

        mins = candidates.min(axis=0)
        maxs = candidates.max(axis=0)
        target_xy = mins + unit_samples * (maxs - mins)

        cost_matrix = cdist(target_xy, candidates)
        _, chosen = linear_sum_assignment(cost_matrix)
    else:  # "spread"
        chosen = [int(rng.integers(len(candidates)))]
        distances = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
        for _ in range(cfg.n_sensors - 1):
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
            distances = np.minimum(
                distances, np.sum((candidates - candidates[chosen[-1]]) ** 2, axis=1)
            )

    xy = candidates[chosen]
    _, z = grid.lookup(xy)
    sensors = np.column_stack((xy, z + cfg.sensor_height_m)).astype(np.float32)
    k = min(max(0, cfg.graph_neighbors), cfg.n_sensors - 1)
    edges = np.empty((2, 0), dtype=np.int64)
    if k:
        _, nearest = cKDTree(sensors[:, :2]).query(sensors[:, :2], k=k + 1)
        src, dst = np.repeat(np.arange(cfg.n_sensors), k), nearest[:, 1:].ravel()
        pairs = np.vstack((np.column_stack((src, dst)), np.column_stack((dst, src))))
        edges = np.unique(pairs, axis=0).T.astype(np.int64)
    return sensors, edges


def setup_sensors_per_episode(
    grids: list[WalkableMap],
    cfg: DataConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    modes = cfg.sensor_placements
    episode_modes = [modes[e % len(modes)] for e in rng.permutation(cfg.episodes)]

    layouts = [
        sample_sensor_layout(grids[e], cfg, rng, placement_mode=episode_modes[e])
        for e in range(cfg.episodes)
    ]
    sensors = np.stack([s for s, _ in layouts])
    edges = [edge for _, edge in layouts]
    return sensors, edges, episode_modes


def compute_edge_attributes(sensors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    src, dst = edges
    delta = sensors[dst] - sensors[src]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.hstack([delta, dist, np.exp(-dist)]).astype(np.float32)


def _path_statistics(paths, n_rx: int, n_tx: int):
    """Reduce per-path RT tensors on the active Dr.Jit backend; transfer only (n_rx, n_tx)."""
    valid = paths.valid
    shape = tuple(valid.shape)
    if len(shape) != 3 or shape[:2] != (n_rx, n_tx):
        raise RuntimeError(f"Unexpected validity shape: {shape}, expected ({n_rx}, {n_tx}, P)")
    if shape[-1] == 0:
        return (
            np.zeros((n_rx, n_tx), np.float32),
            np.zeros((n_rx, n_tx), bool),
            np.zeros((n_rx, n_tx), np.int32),
        )

    def no_antenna_axes(tensor, dtype):
        if tuple(tensor.shape) == shape:
            return tensor
        if tuple(tensor.shape) != (n_rx, 1, n_tx, 1, shape[-1]):
            raise RuntimeError(f"Unexpected path tensor shape: {tensor.shape}")
        return dr.reshape(dtype, tensor, shape)

    real, imag = (no_antenna_axes(x, mi.TensorXf) for x in paths.a)
    gain = dr.sum(dr.select(valid, real * real + imag * imag, 0.0), axis=-1)

    interactions = paths.interactions
    direct = valid if interactions.shape[0] == 0 else (
        valid & no_antenna_axes(dr.all(interactions == 0, axis=0), mi.TensorXb)
    )
    los = dr.sum(dr.select(direct, 1, 0), axis=-1) > 0
    count = dr.sum(dr.select(valid, 1, 0), axis=-1)
    return tuple(np.asarray(x.numpy()).reshape(n_rx, n_tx) for x in (gain, los, count))


def simulate_links(
    scene: rt.Scene,
    positions: np.ndarray,
    sensors: np.ndarray,
    cfg: DataConfig,
    checkpoint: Path | None = None,
):
    """Simulate (E, T, N) links in ONE PathSolver call per episode via channel reciprocity."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (E, T, 3)")
    E, T, _ = positions.shape
    N = cfg.n_sensors
    if sensors.shape != (E, N, 3) or min(E, T, N) < 1:
        raise ValueError("sensors must have shape (E, N, 3) with positive dimensions")
    if not np.isfinite(positions).all() or not np.isfinite(sensors).all():
        raise ValueError("Non-finite sensor or jammer positions")

    outputs = {
        key: np.zeros((E, T, N), dtype)
        for key, dtype in (("path_gain_linear", np.float32), ("los", bool), ("num_paths", np.int32))
    }
    state = json.dumps({**asdict(cfg), "overwrite": None}, default=str, sort_keys=True)
    first = 0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as saved:
            if str(saved["config_json"]) == state:
                first = int(saved["episodes_done"])
                for key in outputs:
                    outputs[key][:] = saved[key]
        print(f"Checkpoint {checkpoint}: resuming at episode {first} of {E}")

    scene.remove(list(scene.transmitters))
    scene.remove(list(scene.receivers))

    tx_slots = [
        Transmitter(
            name=f"sensor_tx_{i:03d}",
            position=sensors[0, i].tolist(),
            power_dbm=cfg.tx_power_dbm,
        )
        for i in range(N)
    ]
    rx_slots = [
        Receiver(
            name=f"jammer_rx_{t:03d}",
            position=positions[0, t].tolist(),
        )
        for t in range(T)
    ]
    scene.add(tx_slots)
    scene.add(rx_slots)

    solver = PathSolver(deterministic=True)
    start_time = perf_counter()

    try:
        with tqdm(
            total=E,
            initial=first,
            desc=f"Ray tracing reciprocal ({cfg.scene_name})",
            file=sys.stdout,
            dynamic_ncols=True,
        ) as progress:
            for e in range(first, E):
                for tx, pos in zip(tx_slots, sensors[e]):
                    tx.position = mi.Point3f(*map(float, pos))

                for rx, pos in zip(rx_slots, positions[e]):
                    rx.position = mi.Point3f(*map(float, pos))

                paths = solver(
                    scene,
                    max_depth=cfg.max_depth,
                    samples_per_src=cfg.samples_per_src,
                    max_num_paths_per_src=cfg.max_num_paths_per_src,
                    synthetic_array=True,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=False,
                    refraction=False,
                    diffraction=cfg.diffraction,
                    seed=cfg.seed + e,
                )

                gain, los, count = _path_statistics(paths, n_rx=T, n_tx=N)
                outputs["path_gain_linear"][e] = gain
                outputs["los"][e] = los
                outputs["num_paths"][e] = count
                progress.update()

                if checkpoint is not None and (e + 1) % cfg.chunk_episodes == 0 and e + 1 < E:
                    with open(checkpoint.with_suffix(".tmp"), "wb") as file:
                        np.savez(file, config_json=state, episodes_done=e + 1, **outputs)
                    checkpoint.with_suffix(".tmp").replace(checkpoint)
    finally:
        scene.remove([tx.name for tx in tx_slots])
        scene.remove([rx.name for rx in rx_slots])

    elapsed = perf_counter() - start_time
    done_episodes = max(1, E - first)
    print(
        f"Solver time: {elapsed:.2f} s | {1000 * elapsed / (done_episodes * T):.2f} ms/position"
        f" | Calls: {done_episodes} (reciprocal) | Backend: {mi.variant()}"
    )
    return outputs


def output_path(cfg: DataConfig) -> Path:
    name = cfg.output_name or f"{cfg.scene_name}_seed{cfg.seed}.h5"
    if Path(name).name != name or not name.endswith(".h5"):
        raise ValueError("output_name must be an .h5 filename, not a path")
    return cfg.output_dir / name


def save_dataset(
    cfg: DataConfig,
    grids: list[WalkableMap],
    sensors: np.ndarray,
    edges: list[np.ndarray],
    episode_modes: list[str],
    positions: np.ndarray,
    velocities: np.ndarray,
    measurements: dict,
    rng: np.random.Generator,
    split_tag: str = "mixed",
) -> Path:
    """Write compressed HDF5 with episode-specific graphs in packed COO format."""
    E, T, N = cfg.episodes, cfg.n_steps, cfg.n_sensors
    if sensors.shape != (E, N, 3) or positions.shape != (E, T, 3):
        raise ValueError("Unexpected sensor or jammer positions shape")
    if velocities.shape != positions.shape or len(edges) != E:
        raise ValueError("Unexpected velocities or number of graphs")

    edge_arrays = [np.asarray(edge, dtype=np.int64) for edge in edges]
    for edge in edge_arrays:
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("Each edge_index must have shape (2, M_e)")
        if edge.size and (edge.min() < 0 or edge.max() >= N):
            raise ValueError("Graph references an invalid sensor index")

    edge_ptr = np.r_[0, np.cumsum([edge.shape[1] for edge in edge_arrays])].astype(np.int64)
    edge_index = np.concatenate(edge_arrays, axis=1)
    edge_attr = np.concatenate(
        [compute_edge_attributes(sensors[e], edges[e]) for e in range(E)], axis=0
    )

    map_free = np.stack([g.free for g in grids])        # (E, H, W)
    map_street = np.stack([g.street for g in grids])    # (E, H, W)
    map_height = np.stack([g.height_m for g in grids])  # (E, H, W)
    map_x_m = np.stack([g.x_m for g in grids])          # (E, W)
    map_y_m = np.stack([g.y_m for g in grids])          # (E, H)

    perm = rng.permutation(cfg.episodes)
    n_test = (
        min(cfg.episodes - 1, max(1, round(cfg.episodes * cfg.test_fraction)))
        if cfg.episodes > 1 and cfg.test_fraction > 0
        else 0
    )

    datasets = {
        "sensors/positions": sensors,
        "sensors/placement_modes": np.asarray(episode_modes, dtype=h5py.string_dtype("utf-8")),
        "graph/edge_index": edge_index,
        "graph/edge_attr": edge_attr,
        "graph/edge_ptr": edge_ptr,
        "jammer/positions": positions,
        "jammer/velocities": velocities,
        "jammer/time_s": np.arange(T, dtype=np.float32) * cfg.dt_s,
        **{f"measurements/{name}": data for name, data in measurements.items()},
        "map/free": map_free,
        "map/street": map_street,
        "map/surface_height_m": map_height,
        "map/x_m": map_x_m,
        "map/y_m": map_y_m,
        "split/train_episodes": np.sort(perm[n_test:]),
        "split/test_episodes": np.sort(perm[:n_test]),
    }

    output = output_path(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output} already exists (use --overwrite to replace it)")
    partial = output.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as file:
        cfg_dict = asdict(cfg)
        cfg_dict["sensor_placements"] = list(cfg.sensor_placements)
        file.attrs["config_json"] = json.dumps(cfg_dict, default=str, sort_keys=True)
        for key, value in cfg_dict.items():
            if isinstance(value, Path):
                file.attrs[key] = str(value)
            elif isinstance(value, (list, tuple)):
                file.attrs[key] = json.dumps(value)
            else:
                file.attrs[key] = value
        file.attrs.update(
            split_tag=split_tag,
            axis_order="episode,time_step,sensor",
            graph_format="packed_coo_with_edge_ptr",
            rssi_model="incoherent_path_power_sum",
            rt_reciprocity="sensors_tx_jammer_rx",
            mitsuba_variant=mi.variant(),
            sionna_rt_version=getattr(rt, "__version__", "unknown"),
        )
        for name, value in datasets.items():
            chunks = None
            if name in ("sensors/positions", "jammer/positions", "jammer/velocities") or name.startswith("measurements/"):
                chunks = (min(E, cfg.chunk_episodes), *value.shape[1:])
            create_kwargs = {"chunks": chunks} if chunks else {}
            if value.dtype.kind != "O":
                create_kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
            file.create_dataset(name, data=value, **create_kwargs)
        file["graph/edge_index"].attrs["axes"] = "source_or_target,edge"
        file["graph/edge_attr"].attrs["columns"] = "dx_m,dy_m,dz_m,distance_m,exp_weight"
        file["graph/edge_ptr"].attrs["description"] = "Edges for episode e: edge_ptr[e]:edge_ptr[e+1]"
    partial.replace(output)
    return output


def parse_config(argv=None) -> DataConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, help="Optional JSON file containing DataConfig fields")
    defaults = DataConfig()
    valid_placements = get_args(PlacementType)

    for field in fields(defaults):
        value = getattr(defaults, field.name)
        opts = {"default": None, "help": f"Default: {value}"}
        if isinstance(value, bool):
            opts["action"] = argparse.BooleanOptionalAction
        elif field.name == "sensor_placements":
            opts.update(type=str, nargs="+", choices=valid_placements)
        elif isinstance(value, tuple):
            opts.update(type=float, nargs=len(value), metavar=("X", "Y"))
        else:
            opts["type"] = type(value)
            if choices := get_args(field.type):
                opts["choices"] = choices
        parser.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name, **opts)

    args = vars(parser.parse_args(argv))
    config_file = args.pop("config")
    data = {}
    if config_file:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            parser.error("JSON configuration must be an object")
        unknown = set(data) - {field.name for field in fields(defaults)}
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")

    data.update({name: value for name, value in args.items() if value is not None})
    for path_key in ("output_dir", "baked_scenes_dir"):
        if path_key in data:
            data[path_key] = Path(data[path_key])
    if "roi_center_xy_m" in data:
        data["roi_center_xy_m"] = tuple(data["roi_center_xy_m"])
    if "sensor_placements" in data:
        data["sensor_placements"] = tuple(data["sensor_placements"])
    cfg = DataConfig(**data)

    if (
        min(cfg.episodes, cfg.n_steps, cfg.n_sensors, cfg.batch_size, cfg.samples_per_src, cfg.max_num_paths_per_src, cfg.chunk_episodes) < 1
        or min(cfg.frequency_hz, cfg.dt_s, cfg.roi_half_size_m, cfg.grid_resolution_m) <= 0
        or cfg.max_depth < 0
        or cfg.graph_neighbors < 0
        or cfg.erosion_iterations < 0
        or cfg.speed_mps < 0
        or cfg.rssi_noise_std_db < 0
        or not 0 <= cfg.test_fraction < 1
        or not cfg.scene_name.strip()
        or not cfg.sensor_placements
        or any(p not in valid_placements for p in cfg.sensor_placements)
        or cfg.roi_mode not in get_args(DataConfig.__dataclass_fields__["roi_mode"].type)
    ):
        parser.error("Invalid configuration values; check ranges and scene/placement choices")
    output_path(cfg)
    return cfg


def compute_wcl_position(sensors: np.ndarray, rssi: np.ndarray, detected: np.ndarray) -> np.ndarray:
    power_linear = np.where(detected, 10.0 ** (rssi / 10.0), 0.0)
    total_power = np.sum(power_linear, axis=-1, keepdims=True)

    weights = np.divide(power_linear, total_power, out=np.zeros_like(power_linear), where=total_power > 0)
    wcl_pos = np.sum(weights[..., None] * sensors[:, None, :, :], axis=2)

    no_detection = total_power.squeeze(-1) == 0
    if np.any(no_detection):
        geom_center = np.mean(sensors, axis=1)
        ep_idx, _ = np.nonzero(no_detection)
        wcl_pos[no_detection] = geom_center[ep_idx]

    return wcl_pos.astype(np.float32)


def add_measurements_with_mask(
    links: dict,
    sensors: np.ndarray,
    edges: list[np.ndarray],
    cfg: DataConfig,
    rng: np.random.Generator,
) -> dict:
    E, T, N = links["path_gain_linear"].shape
    gain = links["path_gain_linear"]
    gain_db = np.full(gain.shape, np.nan, dtype=np.float32)
    np.log10(gain, where=gain > 0, out=gain_db)
    gain_db *= 10.0

    rssi = cfg.tx_power_dbm + gain_db
    if cfg.rssi_noise_std_db > 0:
        rssi += rng.normal(0, cfg.rssi_noise_std_db, size=gain.shape)

    detected_mask = (np.isfinite(rssi) & (rssi >= cfg.sensitivity_dbm)).astype(np.uint8)
    rssi_dbm_nan = np.where(detected_mask == 1, rssi, np.nan).astype(np.float32)
    rssi_dbm_clamped = np.where(detected_mask == 1, rssi, cfg.sensitivity_dbm).astype(np.float32)

    wcl_positions = compute_wcl_position(sensors, rssi_dbm_clamped, detected_mask.astype(bool))
    dist_to_wcl = np.linalg.norm(
        sensors[:, None, :, :] - wcl_positions[:, :, None, :], axis=-1
    ).astype(np.float32)

    rssi_diff_neighbor = np.zeros((E, T, N), dtype=np.float32)
    for e in range(E):
        edge_idx = edges[e]
        if edge_idx.size > 0:
            src, dst = edge_idx[0], edge_idx[1]
            neighbor_means = np.zeros((T, N), dtype=np.float32)
            counts = np.zeros(N, dtype=np.float32)
            for s, d in zip(src, dst):
                neighbor_means[:, s] += rssi_dbm_clamped[e, :, d]
                counts[s] += 1.0
            counts[counts == 0] = 1.0
            neighbor_means /= counts
            rssi_diff_neighbor[e] = rssi_dbm_clamped[e] - neighbor_means

    rssi_time_gradient = np.zeros((E, T, N), dtype=np.float32)
    rssi_time_gradient[:, 1:, :] = np.diff(rssi_dbm_clamped, axis=1)

    return {
        **links,
        "path_gain_db": gain_db,
        "rssi_dbm": rssi_dbm_nan,
        "rssi_dbm_clamped": rssi_dbm_clamped,
        "detected_mask": detected_mask,
        "wcl_positions": wcl_positions,
        "dist_to_wcl": dist_to_wcl,
        "rssi_diff_neighbor": rssi_diff_neighbor,
        "rssi_time_gradient": rssi_time_gradient,
    }


def generate_scene_dataset(cfg: DataConfig, split_tag: str = "mixed") -> Path:
    rng_env = np.random.default_rng(cfg.seed)
    scene, scene_meta, precomputed_map = load_scene_and_walkable_map(cfg)

    grids = [
        sample_episode_grid(
            scene, cfg, rng_env, scene_meta=scene_meta, precomputed_map=precomputed_map
        )
        for _ in range(cfg.episodes)
    ]

    rng_traj = np.random.default_rng(cfg.seed + 1)
    rng_sens = np.random.default_rng(cfg.seed)
    rng_meas = np.random.default_rng(cfg.seed + 2)
    rng_save = np.random.default_rng(cfg.seed + 3)

    positions, velocities = generate_trajectories(grids, cfg, rng_traj)
    sensors, edges, episode_modes = setup_sensors_per_episode(grids, cfg, rng_sens)
    checkpoint = output_path(cfg).with_suffix(".links.npz")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    links = simulate_links(scene, positions, sensors, cfg, checkpoint)
    measurements = add_measurements_with_mask(links, sensors, edges, cfg, rng_meas)

    output = save_dataset(
        cfg, grids, sensors, edges, episode_modes, positions, velocities, measurements, rng_save, split_tag=split_tag
    )
    checkpoint.unlink(missing_ok=True)
    print(f"Saved [{split_tag}]: {output}")
    return output


def main(argv=None) -> Path:
    cfg = parse_config(argv)
    output = output_path(cfg)
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output} already exists (use --overwrite to replace it)")
    return generate_scene_dataset(cfg)


def generate_multi_scene_dataset(
    train_scenes: list[str] = ["würzburg", "nürnberg", "stuttgart", "münchen", "frankfurt"],
    in_dist_test_scenes: list[str] | None = None,
    ood_test_scenes: list[str] = ["vilnius", "kaunas"],
    train_episodes_per_scene: int = 200,
    test_episodes_per_scene: int = 50,
    output_dir: str = "data/multi_scene_ds",
    roi_mode: Literal["manual", "auto", "random"] = "random",
):
    out_path = Path(output_dir)
    if in_dist_test_scenes is None:
        in_dist_test_scenes = list(train_scenes)

    jobs: list[tuple[str, str, int, float, int]] = []
    for idx, scene_name in enumerate(train_scenes):
        jobs.append((scene_name, "train", train_episodes_per_scene, 0.0, 10_000 + idx * 100))
    for idx, scene_name in enumerate(in_dist_test_scenes):
        jobs.append((scene_name, "test_id", test_episodes_per_scene, 0.0, 20_000 + idx * 100))
    for idx, scene_name in enumerate(ood_test_scenes):
        jobs.append((scene_name, "test_ood", test_episodes_per_scene, 0.0, 30_000 + idx * 100))

    print(f"Train scenes ({len(train_scenes)}):          {train_scenes}")
    print(f"In-Dist Test scenes ({len(in_dist_test_scenes)}):  {in_dist_test_scenes}")
    print(f"OOD Test scenes ({len(ood_test_scenes)}):      {ood_test_scenes}")

    for scene_name, split_tag, n_episodes, test_frac, seed in jobs:
        split_dir = out_path / split_tag
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Processing Scene: {scene_name} | Split: {split_tag} ({n_episodes} episodes) ---")

        cfg = DataConfig(
            scene_name=scene_name,
            frequency_hz=1.57542e9,
            tx_power_dbm=20.0,
            sensitivity_dbm=-110.0,
            rssi_noise_std_db=0.0,
            max_depth=3,
            diffraction=True,
            samples_per_src=20_000,
            max_num_paths_per_src=250_000,
            roi_center_xy_m=(0.0, 0.0),
            roi_half_size_m=500.0,
            roi_mode=roi_mode,
            grid_resolution_m=2.0,
            ground_tolerance_m=0.5,
            erosion_iterations=1,
            include_parks_as_walkable=True,
            prefer_streets_for_jammer=True,
            n_sensors=20,
            sensor_height_m=1.5,
            jammer_height_m=1.5,
            sensor_placements=("spread", "random", "lhs"),
            graph_neighbors=3,
            episodes=n_episodes,
            n_steps=60,
            dt_s=1.0,
            speed_mps=5.0,
            heading_std_rad=0.25,
            seed=seed,
            test_fraction=test_frac,
            output_dir=split_dir,
            output_name=f"{scene_name}_{split_tag}.h5",
            overwrite=True,
            debug=False,
            chunk_episodes=64,
        )

        generate_scene_dataset(cfg, split_tag=split_tag)
        dr.sync_thread()
        dr.drjit_malloc_trim()
        gc.collect()


if __name__ == "__main__":
    main()