"""Episode-specific sensor layouts and batched Sionna RT measurements.

python sionna_dataset.py \
    --config dataset_config.example.json \
    --episodes 500 \
    --batch-size 16 \
    --output-name experiment_002.h5
"""

import argparse
import json
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
from scipy.spatial import cKDTree
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from tqdm import tqdm
import sys
import gc

@dataclass(frozen=True)
class DataConfig:
    scene_name: Literal["munich", "etoile", "florence", "san_francisco"] = "etoile"
    frequency_hz: float = 3.5e9
    tx_power_dbm: float = 20.0
    sensitivity_dbm: float = -110.0
    rssi_noise_std_db: float = 0.0
    max_depth: int = 3
    samples_per_src: int = 20_000
    batch_size: int = 8
    roi_center_xy_m: tuple[float, float] = (0.0, 0.0)
    roi_half_size_m: float = 150.0
    grid_resolution_m: float = 3.0
    ground_tolerance_m: float = 0.5
    erosion_iterations: int = 1
    n_sensors: int = 20
    sensor_height_m: float = 1.5
    jammer_height_m: float = 1.5
    sensor_placement: Literal["random", "spread"] = "spread"
    graph_neighbors: int = 3
    episodes: int = 32
    n_steps: int = 60
    dt_s: float = 1.0
    speed_mps: float = 1.5
    heading_std_rad: float = 0.25
    seed: int = 42
    test_fraction: float = 0.2
    output_dir: Path = Path("data/generated")
    output_name: str = ""  # Empty: <scene>_seed<seed>.h5
    overwrite: bool = False
    debug: bool = False
    chunk_episodes: int = 64
    auto_center_roi: bool = True


@dataclass
class WalkableMap:
    x_m: np.ndarray
    y_m: np.ndarray
    height_m: np.ndarray
    free: np.ndarray
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

    def candidates(self) -> np.ndarray:
        iy, ix = np.nonzero(self.free)
        return np.column_stack((self.x_m[ix], self.y_m[iy]))


def load_environment(cfg: DataConfig) -> tuple[rt.Scene, WalkableMap]:
    scene = load_scene(getattr(rt.scene, cfg.scene_name), merge_shapes=True)
    scene.frequency = cfg.frequency_hz
    scene.tx_array = scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    bbox = scene.mi_scene.bbox()
    center = cfg.roi_center_xy_m
    if cfg.auto_center_roi:
        center = ((float(bbox.min.x) + float(bbox.max.x)) / 2,
                  (float(bbox.min.y) + float(bbox.max.y)) / 2)
    n = int(np.floor(2 * cfg.roi_half_size_m / cfg.grid_resolution_m))
    if n < 2:
        raise ValueError("ROI too small for grid resolution")
    offsets = -cfg.roi_half_size_m + (np.arange(n) + 0.5) * cfg.grid_resolution_m
    x_m, y_m = offsets + center[0], offsets + center[1]
    xx, yy = np.meshgrid(x_m, y_m)

    z_top = float(bbox.max.z) + 10
    origins = mi.Point3f(xx.ravel().astype(np.float32), yy.ravel().astype(np.float32),
                         np.full(xx.size, z_top, dtype=np.float32))
    hits = scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0, 0, -1)))
    valid = np.asarray(hits.is_valid().numpy()).reshape(xx.shape)
    height = np.where(valid, z_top - np.asarray(hits.t.numpy()).reshape(xx.shape), np.nan)

    ground = valid & (np.abs(height - float(bbox.min.z)) <= cfg.ground_tolerance_m)
    if cfg.erosion_iterations:
        ground = binary_erosion(ground, iterations=cfg.erosion_iterations)
    components, count = label(ground)
    if not count:
        raise ValueError("No walkable ground found; adjust ROI / ground tolerance")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return scene, WalkableMap(x_m, y_m, height, components == sizes.argmax(), cfg.grid_resolution_m)


def generate_trajectories(grid: WalkableMap, cfg: DataConfig, rng: np.random.Generator):
    positions = np.empty((cfg.episodes, cfg.n_steps, 3), dtype=np.float32)
    candidates = grid.candidates()
    step = cfg.speed_mps * cfg.dt_s
    checks = max(3, int(np.ceil(step / (cfg.grid_resolution_m / 2))) + 1)

    for e in range(cfg.episodes):
        xy = candidates[rng.integers(len(candidates))].copy()
        heading = rng.uniform(0, 2 * np.pi)
        for t in range(cfg.n_steps):
            _, z = grid.lookup(xy)
            positions[e, t] = (*xy, z[0] + cfg.jammer_height_m)
            if t == cfg.n_steps - 1 or step == 0:
                continue
            for attempt in range(32):
                angle = (heading + rng.normal(0, cfg.heading_std_rad)
                         if attempt == 0 else rng.uniform(0, 2 * np.pi))
                move = step * np.array([np.cos(angle), np.sin(angle)])
                samples = xy + np.linspace(0, 1, checks)[:, None] * move
                if grid.lookup(samples)[0].all():
                    xy, heading = xy + move, angle
                    break

    velocity = np.zeros_like(positions)
    velocity[:, 1:] = np.diff(positions, axis=1) / cfg.dt_s
    return positions, velocity


def sample_sensor_layout(grid: WalkableMap, cfg: DataConfig, rng: np.random.Generator):
    candidates = grid.candidates()
    if not 1 <= cfg.n_sensors <= len(candidates):
        raise ValueError("Invalid n_sensors")

    if cfg.sensor_placement == "random":
        chosen = rng.choice(len(candidates), cfg.n_sensors, replace=False)
    else:
        chosen = [int(rng.integers(len(candidates)))]
        distances = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
        for _ in range(cfg.n_sensors - 1):
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
            distances = np.minimum(distances, np.sum(
                (candidates - candidates[chosen[-1]]) ** 2, axis=1
            ))

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


def setup_sensors_per_episode(scene: rt.Scene, grid: WalkableMap, cfg: DataConfig, rng):
    layouts = [sample_sensor_layout(grid, cfg, rng) for _ in range(cfg.episodes)]
    sensors = np.stack([s for s, _ in layouts])
    edges = [edge for _, edge in layouts]
    for i, pos in enumerate(sensors[0]):
        scene.add(Receiver(name=f"sensor_{i:03d}", position=pos.tolist()))
    return sensors, edges


def compute_edge_attributes(sensors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    src, dst = edges
    delta = sensors[dst] - sensors[src]
    #return np.column_stack((delta, np.linalg.norm(delta, axis=1))).astype(np.float32)
    
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.hstack([delta, dist, np.exp(-dist)]).astype(np.float32)
    


def _path_statistics(paths, n_rx: int, batch_size: int):
    """Reduce per-path RT tensors on the active Dr.Jit backend; transfer only (N, B)."""
    valid = paths.valid  # (N, B, P) for synthetic single-antenna arrays
    shape = tuple(valid.shape)
    if len(shape) != 3 or shape[:2] != (n_rx, batch_size):
        raise RuntimeError(f"Unexpected validity shape: {shape}")
    if shape[-1] == 0:
        return (np.zeros((n_rx, batch_size), np.float32),
                np.zeros((n_rx, batch_size), bool),
                np.zeros((n_rx, batch_size), np.int32))

    def no_antenna_axes(tensor, dtype):
        if tuple(tensor.shape) == shape:
            return tensor
        if tuple(tensor.shape) != (n_rx, 1, batch_size, 1, shape[-1]):
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
    return tuple(np.asarray(x.numpy()).reshape(n_rx, batch_size) for x in (gain, los, count))


def simulate_links(scene: rt.Scene, positions: np.ndarray, sensors: np.ndarray, cfg: DataConfig):
    """Simulate (E, T, N) links with one solver and reusable TX slots."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (E, T, 3)")
    E, T, _ = positions.shape
    if sensors.shape != (E, cfg.n_sensors, 3) or min(E, T, cfg.n_sensors) < 1:
        raise ValueError("sensors must have shape (E, N, 3) with positive dimensions")
    if cfg.batch_size < 1 or not np.isfinite(positions).all() or not np.isfinite(sensors).all():
        raise ValueError("Invalid batch size or non-finite positions")

    N, B = cfg.n_sensors, min(cfg.batch_size, T)
    names = [f"sensor_{i:03d}" for i in range(N)]
    if list(scene.receivers) != names:
        raise ValueError("Unexpected receiver ordering")
    receivers = [scene.receivers[name] for name in names]
    outputs = {key: np.zeros((E, T, N), dtype) for key, dtype in (
        ("path_gain_linear", np.float32), ("los", bool), ("num_paths", np.int32)
    )}

    scene.remove(list(scene.transmitters))
    slots = [Transmitter(name=f"jammer_slot_{i:03d}",
                         position=positions[0, 0].tolist(), power_dbm=cfg.tx_power_dbm)
             for i in range(B)]
    scene.add(slots)
    solver = PathSolver(deterministic=True)
    calls = E * ((T + B - 1) // B)
    start_time = perf_counter()

    try:
        with tqdm(total=calls, desc=f"Ray tracing ({scene.name if hasattr(scene, 'name') else 'Scene'})", file=sys.stdout, dynamic_ncols=True) as progress:
            #with tqdm(total=calls, desc="Ray tracing") as progress:
            for e in range(E):
                for receiver, pos in zip(receivers, sensors[e]):
                    receiver.position = mi.Point3f(*map(float, pos))

                for start in range(0, T, B):
                    batch = positions[e, start:start + B]
                    for i, slot in enumerate(slots):
                        slot.position = mi.Point3f(*map(float, batch[min(i, len(batch) - 1)]))
                    paths = solver(
                        scene, max_depth=cfg.max_depth, samples_per_src=cfg.samples_per_src,
                        synthetic_array=True, los=True, specular_reflection=True,
                        diffuse_reflection=False, refraction=False, diffraction=False,
                        seed=cfg.seed + e * T + start,
                    )
                    gain, los, count = _path_statistics(paths, N, B)
                    end = start + len(batch)
                    outputs["path_gain_linear"][e, start:end] = gain.T[:len(batch)]
                    outputs["los"][e, start:end] = los.T[:len(batch)]
                    outputs["num_paths"][e, start:end] = count.T[:len(batch)]
                    progress.update()
    finally:
        scene.remove([slot.name for slot in slots])

    elapsed = perf_counter() - start_time
    print(f"Solver time: {elapsed:.2f} s | {1000 * elapsed / (E * T):.2f} ms/position"
          f" | Calls: {calls} | Backend: {mi.variant()}")
    return outputs


def add_measurements(links: dict, cfg: DataConfig, rng: np.random.Generator) -> dict:
    gain = links["path_gain_linear"]
    gain_db = np.full(gain.shape, np.nan, dtype=np.float32)
    np.log10(gain, where=gain > 0, out=gain_db)
    gain_db *= 10
    rssi = cfg.tx_power_dbm + gain_db
    if cfg.rssi_noise_std_db > 0:
        rssi += rng.normal(0, cfg.rssi_noise_std_db, size=gain.shape)
    detected = np.isfinite(rssi) & (rssi >= cfg.sensitivity_dbm)
    return {**links, "path_gain_db": gain_db,
            "rssi_dbm": np.where(detected, rssi, np.nan).astype(np.float32),
            "detected": detected}



def output_path(cfg: DataConfig) -> Path:
    """Resolve the dataset filename, without creating any directories."""
    name = cfg.output_name or f"{cfg.scene_name}_seed{cfg.seed}.h5"
    if Path(name).name != name or not name.endswith(".h5"):
        raise ValueError("output_name must be an .h5 filename, not a path")
    return cfg.output_dir / name


def save_dataset(
    cfg: DataConfig,
    grid: WalkableMap,
    sensors: np.ndarray,
    edges: list[np.ndarray],
    positions: np.ndarray,
    velocities: np.ndarray,
    measurements: dict,
    rng: np.random.Generator,
) -> Path:
    """Write compressed HDF5 with episode-specific graphs in packed COO format."""
    E, T, N = cfg.episodes, cfg.n_steps, cfg.n_sensors
    if sensors.shape != (E, N, 3) or positions.shape != (E, T, 3):
        raise ValueError("Unexpected sensor or jammer positions shape")
    if velocities.shape != positions.shape or len(edges) != E:
        raise ValueError("Unexpected velocities or number of graphs")
    if any(np.shape(value) != (E, T, N) for value in measurements.values()):
        raise ValueError("Measurements must have shape (E, T, N)")

    # Store variable-size graphs as a single concatenated edge list.
    edge_arrays = [np.asarray(edge, dtype=np.int64) for edge in edges]
    for edge in edge_arrays:
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("Each edge_index must have shape (2, M_e)")
        if edge.size and (edge.min() < 0 or edge.max() >= N):
            raise ValueError("Graph references an invalid sensor index")

    edge_ptr = np.r_[0, np.cumsum([edge.shape[1] for edge in edge_arrays])].astype(np.int64)
    edge_index = np.concatenate(edge_arrays, axis=1)
    edge_attr = np.concatenate([
        compute_edge_attributes(sensors[e], edges[e]) for e in range(E)
    ], axis=0)

    perm = rng.permutation(E)
    n_test = min(E - 1, max(1, round(E * cfg.test_fraction))) if E > 1 and cfg.test_fraction > 0 else 0
    
    datasets = {
        "sensors/positions": sensors,
        "graph/edge_index": edge_index,
        "graph/edge_attr": edge_attr,
        "graph/edge_ptr": edge_ptr,
        "jammer/positions": positions,
        "jammer/velocities": velocities,
        "jammer/time_s": np.arange(T, dtype=np.float32) * cfg.dt_s,
        **{f"measurements/{name}": data for name, data in measurements.items()},
        "map/free": grid.free,
        "map/surface_height_m": grid.height_m,
        "map/x_m": grid.x_m,
        "map/y_m": grid.y_m,
        "split/train_episodes": np.sort(perm[n_test:]),
        "split/test_episodes": np.sort(perm[:n_test]),
    }

    output = output_path(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 'x' refuses to overwrite an existing file, even if another process created it.
    with h5py.File(output, "w" if cfg.overwrite else "x") as file:
        file.attrs["config_json"] = json.dumps(asdict(cfg), default=str, sort_keys=True)
        for key, value in asdict(cfg).items():
            file.attrs[key] = str(value) if isinstance(value, Path) else value
        file.attrs.update(
            axis_order="episode,time_step,sensor",
            graph_format="packed_coo_with_edge_ptr",
            rssi_model="incoherent_path_power_sum",
            mitsuba_variant=mi.variant(),
            sionna_rt_version=getattr(rt, "__version__", "unknown"),
        )
        for name, value in datasets.items():
            # Chunk temporal tensors by episodes to support partial reads.
            chunks = None
            if name in ("sensors/positions", "jammer/positions", "jammer/velocities") or name.startswith("measurements/"):
                chunks = (min(E, cfg.chunk_episodes), *value.shape[1:])
            file.create_dataset(
                name, data=value, compression="gzip", compression_opts=4,
                shuffle=True, **({"chunks": chunks} if chunks else {}),
            )
        file["graph/edge_index"].attrs["axes"] = "source_or_target,edge"
        file["graph/edge_attr"].attrs["columns"] = "dx_m,dy_m,dz_m,distance_m,exp_weight"
        file["graph/edge_ptr"].attrs["description"] = "Edges for episode e: edge_ptr[e]:edge_ptr[e+1]"
    return output


def parse_config(argv=None) -> DataConfig:
    """Read optional JSON config, then apply CLI overrides for all DataConfig fields."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, help="Optional JSON file containing DataConfig fields")
    defaults = DataConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        opts = {"default": None, "help": f"Default: {value}"}
        if isinstance(value, bool):
            opts["action"] = argparse.BooleanOptionalAction
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
    if "output_dir" in data:
        data["output_dir"] = Path(data["output_dir"])
    if "roi_center_xy_m" in data:
        data["roi_center_xy_m"] = tuple(data["roi_center_xy_m"])
    cfg = DataConfig(**data)

    if (min(cfg.episodes, cfg.n_steps, cfg.n_sensors, cfg.batch_size,
            cfg.samples_per_src, cfg.chunk_episodes) < 1
        or min(cfg.frequency_hz, cfg.dt_s, cfg.roi_half_size_m,
               cfg.grid_resolution_m) <= 0
        or cfg.max_depth < 0 or cfg.graph_neighbors < 0 or cfg.erosion_iterations < 0
        or cfg.speed_mps < 0 or cfg.rssi_noise_std_db < 0
        or not 0 <= cfg.test_fraction < 1
        or cfg.scene_name not in get_args(DataConfig.__dataclass_fields__["scene_name"].type)
        or cfg.sensor_placement not in ("random", "spread")):
        parser.error("Invalid configuration values; check ranges and scene/placement choices")
    output_path(cfg)  # Validate the output filename before starting ray tracing.
    return cfg


def main(argv=None) -> Path:
    cfg = parse_config(argv)
    output = output_path(cfg)
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output} already exists (use --overwrite to replace it)")

    scene, grid = load_environment(cfg)
    positions, velocities = generate_trajectories(grid, cfg, np.random.default_rng(cfg.seed + 1))
    sensors, edges = setup_sensors_per_episode(scene, grid, cfg, np.random.default_rng(cfg.seed))
    links = simulate_links(scene, positions, sensors, cfg)
    measurements = add_measurements(links, cfg, np.random.default_rng(cfg.seed + 2))
    output = save_dataset(
        cfg, grid, sensors, edges, positions, velocities, measurements,
        np.random.default_rng(cfg.seed + 3),
    )
    print(f"Saved {output.resolve()} | sensors={sensors.shape} | RSSI={measurements['rssi_dbm'].shape}")
    return output


import json
from dataclasses import asdict
from pathlib import Path
import h5py
import numpy as np

def compute_wcl_position(sensors: np.ndarray, rssi: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """ Calc. WCL per Episode and TimeStep """
    E, T, N = rssi.shape
    wcl_pos = np.zeros((E, T, 3), dtype=np.float32)

    power_linear = np.where(detected, 10.0 ** (rssi / 10.0), 0.0)
    total_power = np.sum(power_linear, axis=-1, keepdims=True)  # (E, T, 1)

    # Gewichtung pro Sensor berechnen
    weights = np.divide(power_linear, total_power, out=np.zeros_like(power_linear), where=total_power > 0)

    # 3D-Schwerpunkt berechnen: sum_i(w_i * sensor_pos_i)
    wcl_pos = np.sum(weights[..., None] * sensors[:, None, :, :], axis=2)

    # Fallback: Falls in einem Zeitschritt kein Sensor ein Signal empfängt, geometrischen Mittelpunkt nutzen
    no_detection = (total_power.squeeze(-1) == 0)
    if np.any(no_detection):
        geom_center = np.mean(sensors, axis=1)  # (E, 3)
        wcl_pos[no_detection] = geom_center[no_detection[:, 0]]

    return wcl_pos.astype(np.float32)

def add_measurements_with_mask(links: dict, sensors: np.ndarray, edges: list[np.ndarray], cfg: DataConfig, rng: np.random.Generator) -> dict:
    """Computes RSSI values and generates an explicit binary mask array, WCL prior and local/timely noise statistics"""
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

    wcl_positions = compute_wcl_position(sensors, rssi_dbm_clamped, detected_mask.astype(bool)) # (E, T, 3)
    dist_to_wcl = np.linalg.norm(sensors[:, None, :, :] - wcl_positions[:, :, None, :], axis=-1).astype(np.float32) # Euclidean dist to WCL

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

    # Signal-Gradient: Delta RSSI (t) - RSSI (t-1)
    rssi_time_gradient = np.zeros((E, T, N), dtype=np.float32)
    rssi_time_gradient[:, 1:, :] = np.diff(rssi_dbm_clamped, axis=1)

    return {
        **links,
        "path_gain_db": gain_db,
        "rssi_dbm": rssi_dbm_nan,
        "rssi_dbm_clamped": rssi_dbm_clamped,
        "detected_mask": detected_mask,
        # Neue erweiterte Features:
        "wcl_positions": wcl_positions,
        "dist_to_wcl": dist_to_wcl, # Sensor-Distance to WCL (E, T, N)
        "rssi_diff_neighbor": rssi_diff_neighbor,
        "rssi_time_gradient": rssi_time_gradient,
    }

def generate_multi_scene_dataset(
    train_scenes: list[str] = ["munich", "florence", "san_francisco"],
    test_scene: str = "etoile",
    episodes_per_scene: int = 100,
    output_dir: str = "data/multi_scene_ds"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_scenes = train_scenes + [test_scene]
    print(f"Generating datasets for Train Scenes: {train_scenes}")
    print(f"Generating dataset for Test Scene: {test_scene}")

    for idx, scene_name in enumerate(all_scenes):
        split_type = "test" if scene_name == test_scene else "train"
        print(f"\n--- Processing Scene: {scene_name} ({split_type}) ---")

        cfg = DataConfig(
            scene_name=scene_name,
            episodes=episodes_per_scene,
            output_dir=out_path,
            output_name=f"{scene_name}_{split_type}.h5",
            overwrite=True
        )

        cfg = DataConfig(
            scene_name=scene_name,
            frequency_hz=3.5e9,
            tx_power_dbm=20.0,
            sensitivity_dbm=-110.0,
            rssi_noise_std_db=0.0,
            max_depth=3,
            samples_per_src=20_000,
            batch_size=8,
            roi_center_xy_m=(0.0, 0.0),
            roi_half_size_m=150.0,
            grid_resolution_m=3.0,
            ground_tolerance_m=0.5,
            erosion_iterations=1,
            n_sensors=20,
            sensor_height_m=1.5,
            jammer_height_m=1.5,
            sensor_placement="spread",
            graph_neighbors=3,
            episodes=episodes_per_scene,
            n_steps=60,
            dt_s=1.0,
            speed_mps=1.5,
            heading_std_rad=0.25,
            seed=42 + idx * 100,
            test_fraction=0.2,
            output_dir=out_path,
            output_name=f"{scene_name}_{split_type}.h5",
            overwrite=True,
            debug=False,
            chunk_episodes=64,
            auto_center_roi=True,
        )


        scene, grid = load_environment(cfg)
        rng_traj = np.random.default_rng(cfg.seed + 1)
        rng_sens = np.random.default_rng(cfg.seed)
        rng_meas = np.random.default_rng(cfg.seed + 2)
        rng_save = np.random.default_rng(cfg.seed + 3)

        positions, velocities = generate_trajectories(grid, cfg, rng_traj)
        sensors, edges = setup_sensors_per_episode(scene, grid, cfg, rng_sens)
        links = simulate_links(scene, positions, sensors, cfg)

        measurements = add_measurements_with_mask(links, cfg, rng_meas)

        # Save HDF5 file per scene
        save_dataset(
            cfg, grid, sensors, edges, positions, velocities, measurements, rng_save
        )
        print(f"Saved: {cfg.output_dir / cfg.output_name}")

        del scene, grid, positions, velocities, sensors, links, measurements
        dr.drjit_malloc_trim()  # Frees Dr.Jit memory caches
        gc.collect()

if __name__ == "__main__":
    #main()
    # Execute generation
    generate_multi_scene_dataset(
        train_scenes=["munich", "florence"],
        test_scene="etoile",
        episodes_per_scene=500
    )
