"""
--big-size = 2000, --sub-size = 200
==============================================================================
BENCHMARK RESULTS & COMPUTATIONAL BREAKDOWN
==============================================================================

1. Strategy A (Single 10 km Big Scene)
  - OSM Extract:               7.23 s
  - SceneBaker 3D Mesh:      167.78 s
  - Sionna RT Scene Load:      3.80 s
  - One-time Prep Total:     178.81 s
  - Avg Ray Tracing / ROI:     1.48 s  (in 10 km BVH tree)
  - Total Execution (5 ROIs):  186.22 s

2. Strategy B (Individual 2 km Small Scenes)
  - Avg OSM Extract / ROI:     1.33 s
  - Avg Mesh Bake / ROI:      19.64 s
  - Avg Scene Load / ROI:      0.43 s
  - Avg Prep Total / ROI:     21.41 s
  - Avg Ray Tracing / ROI:     0.03 s  (in 2 km BVH tree)
  - Avg Total / ROI:          21.44 s
  - Total Execution (5 ROIs):  107.21 s

==============================================================================
SCALING ANALYSIS & BREAK-EVEN POINT
==============================================================================

[*] Break-even point (N*): ~9 subregion extractions
    - If extracting < 9 ROIs: Strategy B (direct 2 km scenes) is faster.
    - If extracting >= 9 ROIs: Strategy A (10 km big scene) is faster.

Projections for N ROI Extractions:
  N ROIs |  Strategy A (10km Big) | Strategy B (2km Small) |  Speedup Factor
---------------------------------------------------------------------------
       1 |               180.3 s |                21.4 s |       0.12x (Strategy B)
       5 |               186.2 s |               107.2 s |       0.58x (Strategy B)
      10 |               193.6 s |               214.4 s |       1.11x (Strategy A)
      25 |               215.9 s |               536.1 s |       2.48x (Strategy A)
      50 |               252.9 s |              1072.1 s |       4.24x (Strategy A)
     100 |               327.0 s |              2144.2 s |       6.56x (Strategy A)
     500 |               919.6 s |             10721.1 s |      11.66x (Strategy A)

---

--big-size = 10_000, --sub-size = 2_000
==============================================================================
BENCHMARK RESULTS & COMPUTATIONAL BREAKDOWN
==============================================================================

1. Strategy A (Single 10 km Big Scene)
  - OSM Extract:              12.72 s
  - SceneBaker 3D Mesh:     1628.49 s
  - Sionna RT Scene Load:     14.69 s
  - One-time Prep Total:    1655.90 s
  - Avg Ray Tracing / ROI:     0.58 s  (in 10 km BVH tree)
  - Total Execution (5 ROIs): 1658.81 s

2. Strategy B (Individual 2 km Small Scenes)
  - Avg OSM Extract / ROI:     3.08 s
  - Avg Mesh Bake / ROI:      69.96 s
  - Avg Scene Load / ROI:      1.64 s
  - Avg Prep Total / ROI:     74.67 s
  - Avg Ray Tracing / ROI:     0.02 s  (in 2 km BVH tree)
  - Avg Total / ROI:          74.70 s
  - Total Execution (5 ROIs):  373.49 s

==============================================================================
SCALING ANALYSIS & BREAK-EVEN POINT
==============================================================================

[*] Break-even point (N*): ~23 subregion extractions
    - If extracting < 23 ROIs: Strategy B (direct 2 km scenes) is faster.
    - If extracting >= 23 ROIs: Strategy A (10 km big scene) is faster.

Projections for N ROI Extractions:
  N ROIs |  Strategy A (10km Big) | Strategy B (2km Small) |  Speedup Factor
---------------------------------------------------------------------------
       1 |              1656.5 s |                74.7 s |       0.05x (Strategy B)
       5 |              1658.8 s |               373.5 s |       0.23x (Strategy B)
      10 |              1661.7 s |               747.0 s |       0.45x (Strategy B)
      25 |              1670.5 s |              1867.5 s |       1.12x (Strategy A)
      50 |              1685.0 s |              3734.9 s |       2.22x (Strategy A)
     100 |              1714.1 s |              7469.8 s |       4.36x (Strategy A)
     500 |              1947.2 s |             37349.2 s |      19.18x (Strategy A)

"""

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------
# 1. Environment & Path Setup
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
EXTERNALS_DIR = ROOT_DIR / "externals"

for path in [
    EXTERNALS_DIR / "tools" / "sionna-scene-baker",
    EXTERNALS_DIR / "baseline",
]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scenebaker
from sionna import rt

INPUT_DIR = ROOT_DIR / "workspace_inputs"
TERRAIN_CACHE_DIR = INPUT_DIR / "terrain" / "srtm"
TERRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Geographic & OSM Helpers
# --------------------------------------------------

def compute_bbox(lat: float, lon: float, size_m: float, buffer_m: float = 200.0):
    """Calculates latitude/longitude min/max bounds for a given spatial area."""
    half_range = (size_m / 2.0) + buffer_m
    dlat = half_range / 111_320
    dlon = half_range / (111_320 * np.cos(np.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def extract_osm_for_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    osm_region: str,
    out_osm_file: Path,
):
    """Extracts a bounding-box OSM map from pre-tiled OSM dataset."""
    osm_dir = INPUT_DIR / "osm" / "prepared" / osm_region
    meta_path = osm_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json at {osm_dir}")

    meta = json.loads(meta_path.read_text())
    tile_deg = meta["tile_deg"]
    tile_dir = osm_dir / meta["tile_dir"]
    eps = 1e-10

    lat_start = math.floor(lat_min / tile_deg)
    lat_end = math.floor((lat_max - eps) / tile_deg)
    lon_start = math.floor(lon_min / tile_deg)
    lon_end = math.floor((lon_max - eps) / tile_deg)

    tiles = [
        tile_dir / f"{lat_i}_{lon_i}.osm.pbf"
        for lat_i in range(lat_start, lat_end + 1)
        for lon_i in range(lon_start, lon_end + 1)
    ]

    for tile in tiles:
        if not tile.exists():
            raise FileNotFoundError(f"Missing OSM tile: {tile}")

    out_osm_file.parent.mkdir(parents=True, exist_ok=True)

    if len(tiles) == 1:
        osm_source = tiles[0]
    else:
        osm_source = out_osm_file.parent / "merged.osm.pbf"
        subprocess.run(
            [
                "osmium", "merge",
                *map(str, tiles),
                "-o", str(osm_source),
                "--overwrite",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    subprocess.run(
        [
            "osmium", "extract",
            "-b", f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "-s", "smart",
            "-S", "tags=building,building:part,landuse,natural,water,leisure",
            "--set-bounds", str(osm_source),
            "-o", str(out_osm_file),
            "--overwrite",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def generate_subregion_centers(
    center_lat: float,
    center_lon: float,
    big_size_m: float,
    sub_size_m: float,
    num_subregions: int,
):
    """Generates subregion coordinates and meter offsets within a larger region."""
    max_offset = (big_size_m - sub_size_m) / 2.0
    grid_dim = int(np.ceil(np.sqrt(num_subregions)))

    if grid_dim > 1:
        offsets = np.linspace(-max_offset, max_offset, grid_dim)
    else:
        offsets = [0.0]

    centers = []
    for dy_m in offsets:
        for dx_m in offsets:
            if len(centers) >= num_subregions:
                break
            lat_sub = center_lat + (dy_m / 111_320)
            lon_sub = center_lon + (dx_m / (111_320 * np.cos(np.radians(center_lat))))
            centers.append((lat_sub, lon_sub, dx_m, dy_m))
        if len(centers) >= num_subregions:
            break

    return centers


# --------------------------------------------------
# 3. Benchmark Runners
# --------------------------------------------------

def benchmark_strategy_a(
    center_lat: float,
    center_lon: float,
    big_size: float,
    sub_size: float,
    sub_centers: list,
    osm_region: str,
    work_dir: Path,
    grid_res: int,
    samples_per_tx: int,
):
    """Strategy A: Bake ONE big 10km scene, then solve RT on multiple 2km subregions inside it."""
    print(f"\n[+] Strategy A: One Big Scene ({big_size/1000:g} km) -> {len(sub_centers)} Subregions ({sub_size/1000:g} km)")
    t0 = time.perf_counter()

    lat_min, lat_max, lon_min, lon_max = compute_bbox(center_lat, center_lon, big_size)
    osm_file = work_dir / "osm" / "map.osm"
    extract_osm_for_bbox(lat_min, lat_max, lon_min, lon_max, osm_region, osm_file)
    t_osm = time.perf_counter()

    scene = scenebaker.load_scene(
        lat_min, lat_max, lon_min, lon_max,
        terrain=True,
        out_dir=work_dir,
        reuse_osm=True,
        terrain_dir=TERRAIN_CACHE_DIR,
        ground_z=0.0,
        min_ground_half_width=(big_size / 2.0) + 300,
        terrain_grid_size=grid_res,
    )
    t_bake = time.perf_counter()

    ground_center = scenebaker.height(scene, 0, 0, buildings=False) or 0.0
    scene.frequency = 1.57542e9
    #antenna = rt.PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
    antenna = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = antenna
    scene.rx_array = antenna
    t_load = time.perf_counter()

    cell_size = sub_size / grid_res
    rt_times = []

    for idx, (lat_sub, lon_sub, dx_m, dy_m) in enumerate(sub_centers):
        t_sub_start = time.perf_counter()

        tx_name = f"tx_{idx}"
        #ground_z = scenebaker.height(scene, dx_m, dy_m, buildings=False) or ground_center
        #scene.add(rt.Transmitter(name=tx_name, position=[dx_m, dy_m, ground_z + 3.0]))

        #_ = rt.RadioMapSolver()(
        #    scene,
        #    center=[dx_m, dy_m, ground_z + 1.5],
        #    orientation=[0, 0, 0],
        #    size=[sub_size, sub_size],
        #    cell_size=(cell_size, cell_size),
        #    max_depth=3,
        #    samples_per_tx=samples_per_tx,
        #)
        ground_z = float(scenebaker.height(scene, dx_m, dy_m, buildings=False) or ground_center)
        tx_pos = [float(dx_m), float(dy_m), ground_z + 3.0]
        scene.add(rt.Transmitter(name=tx_name, position=tx_pos))

        _ = rt.RadioMapSolver()(
            scene,
            center=[float(dx_m), float(dy_m), ground_z + 1.5],
            orientation=[0.0, 0.0, 0.0],
            size=[float(sub_size), float(sub_size)],
            cell_size=(float(cell_size), float(cell_size)),
            max_depth=3,
            samples_per_tx=samples_per_tx,
        )

        scene.remove(tx_name)
        rt_times.append(time.perf_counter() - t_sub_start)

    del scene
    gc.collect()

    return {
        "osm_time": t_osm - t0,
        "bake_time": t_bake - t_osm,
        "load_time": t_load - t_bake,
        "total_prep": t_load - t0,
        "rt_times": rt_times,
        "avg_rt_per_roi": float(np.mean(rt_times)),
        "total_rt": sum(rt_times),
        "total_time": (t_load - t0) + sum(rt_times),
    }


def benchmark_strategy_b(
    sub_centers: list,
    sub_size: float,
    osm_region: str,
    work_dir: Path,
    grid_res: int,
    samples_per_tx: int,
):
    """Strategy B: Bake individual small 2km scenes on demand for each ROI."""
    print(f"\n[+] Strategy B: Direct Small Scenes ({sub_size/1000:g} km) for {len(sub_centers)} ROIs")

    osm_times, bake_times, load_times, rt_times, total_per_roi = [], [], [], [], []
    cell_size = sub_size / grid_res

    for idx, (lat_sub, lon_sub, _, _) in enumerate(sub_centers):
        t0 = time.perf_counter()
        roi_dir = work_dir / f"roi_{idx:03d}"

        lat_min, lat_max, lon_min, lon_max = compute_bbox(lat_sub, lon_sub, sub_size)
        osm_file = roi_dir / "osm" / "map.osm"
        extract_osm_for_bbox(lat_min, lat_max, lon_min, lon_max, osm_region, osm_file)
        t_osm = time.perf_counter()

        scene = scenebaker.load_scene(
            lat_min, lat_max, lon_min, lon_max,
            terrain=True,
            out_dir=roi_dir,
            reuse_osm=True,
            terrain_dir=TERRAIN_CACHE_DIR,
            ground_z=0.0,
            min_ground_half_width=(sub_size / 2.0) + 200,
            terrain_grid_size=grid_res,
        )
        t_bake = time.perf_counter()

        ground_center = scenebaker.height(scene, 0, 0, buildings=False) or 0.0
        scene.frequency = 1.57542e9
        #antenna = rt.PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        antenna = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        scene.tx_array = antenna
        scene.rx_array = antenna
        #scene.add(rt.Transmitter(name="tx", position=[0, 0, ground_center + 3.0]))

        ground_center = float(scenebaker.height(scene, 0, 0, buildings=False) or 0.0)
        scene.add(rt.Transmitter(name="tx", position=[0.0, 0.0, ground_center + 3.0]))

        t_load = time.perf_counter()

        #_ = rt.RadioMapSolver()(
        #    scene,
        #    center=[0, 0, ground_center + 1.5],
        #    orientation=[0, 0, 0],
        #    size=[sub_size, sub_size],
        #    cell_size=(cell_size, cell_size),
        #    max_depth=3,
        #    samples_per_tx=samples_per_tx,
        #)

        _ = rt.RadioMapSolver()(
            scene,
            center=[0.0, 0.0, ground_center + 1.5],
            orientation=[0.0, 0.0, 0.0],
            size=[float(sub_size), float(sub_size)],
            cell_size=(float(cell_size), float(cell_size)),
            max_depth=3,
            samples_per_tx=samples_per_tx,
        )
        t_rt = time.perf_counter()

        osm_times.append(t_osm - t0)
        bake_times.append(t_bake - t_osm)
        load_times.append(t_load - t_bake)
        rt_times.append(t_rt - t_load)
        total_per_roi.append(t_rt - t0)

        del scene
        gc.collect()

    return {
        "avg_osm": float(np.mean(osm_times)),
        "avg_bake": float(np.mean(bake_times)),
        "avg_load": float(np.mean(load_times)),
        "avg_rt": float(np.mean(rt_times)),
        "avg_prep_per_roi": float(np.mean(osm_times) + np.mean(bake_times) + np.mean(load_times)),
        "avg_total_per_roi": float(np.mean(total_per_roi)),
        "total_osm": sum(osm_times),
        "total_bake": sum(bake_times),
        "total_load": sum(load_times),
        "total_rt": sum(rt_times),
        "total_time": sum(total_per_roi),
    }


# --------------------------------------------------
# 4. Main Execution & Comparison
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Sionna/SceneBaker scene generation strategies.")
    parser.add_argument("--lat", type=float, default=49.7913, help="Center latitude (default: Würzburg)")
    parser.add_argument("--lon", type=float, default=9.9534, help="Center longitude (default: Würzburg)")
    parser.add_argument("--big-size", type=float, default=10_000.0, help="Large scene side length in meters (default: 10000)")
    parser.add_argument("--sub-size", type=float, default=2_000.0, help="Subregion side length in meters (default: 2000)")
    parser.add_argument("--num-subregions", type=int, default=5, help="Number of subregion extractions to test")
    parser.add_argument("--osm-region", type=str, default="germany", help="Prepared OSM region name")
    parser.add_argument("--grid-res", type=int, default=96, help="Grid resolution for radio map solver")
    parser.add_argument("--samples-per-tx", type=int, default=20_000, help="Ray tracing samples per transmitter")
    args = parser.parse_args()

    sub_centers = generate_subregion_centers(
        args.lat, args.lon, args.big_size, args.sub_size, args.num_subregions
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run Strategy A
        res_a = benchmark_strategy_a(
            args.lat, args.lon, args.big_size, args.sub_size, sub_centers,
            args.osm_region, tmp_path / "strat_a", args.grid_res, args.samples_per_tx
        )

        # Run Strategy B
        res_b = benchmark_strategy_b(
            sub_centers, args.sub_size, args.osm_region,
            tmp_path / "strat_b", args.grid_res, args.samples_per_tx
        )

    # --------------------------------------------------
    # Reporting & Break-even analysis
    # --------------------------------------------------
    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS & COMPUTATIONAL BREAKDOWN")
    print("=" * 78)

    print("\n1. Strategy A (Single 10 km Big Scene)")
    print(f"  - OSM Extract:            {res_a['osm_time']:7.2f} s")
    print(f"  - SceneBaker 3D Mesh:     {res_a['bake_time']:7.2f} s")
    print(f"  - Sionna RT Scene Load:   {res_a['load_time']:7.2f} s")
    print(f"  - One-time Prep Total:    {res_a['total_prep']:7.2f} s")
    print(f"  - Avg Ray Tracing / ROI:  {res_a['avg_rt_per_roi']:7.2f} s  (in 10 km BVH tree)")
    print(f"  - Total Execution ({len(sub_centers)} ROIs): {res_a['total_time']:7.2f} s")

    print("\n2. Strategy B (Individual 2 km Small Scenes)")
    print(f"  - Avg OSM Extract / ROI:  {res_b['avg_osm']:7.2f} s")
    print(f"  - Avg Mesh Bake / ROI:    {res_b['avg_bake']:7.2f} s")
    print(f"  - Avg Scene Load / ROI:   {res_b['avg_load']:7.2f} s")
    print(f"  - Avg Prep Total / ROI:   {res_b['avg_prep_per_roi']:7.2f} s")
    print(f"  - Avg Ray Tracing / ROI:  {res_b['avg_rt']:7.2f} s  (in 2 km BVH tree)")
    print(f"  - Avg Total / ROI:        {res_b['avg_total_per_roi']:7.2f} s")
    print(f"  - Total Execution ({len(sub_centers)} ROIs): {res_b['total_time']:7.2f} s")

    # Scaling analysis
    prep_a = res_a['total_prep']
    rt_a = res_a['avg_rt_per_roi']
    prep_b = res_b['avg_prep_per_roi']
    rt_b = res_b['avg_rt']

    delta_prep = prep_b - (prep_a / max(1, len(sub_centers)))
    delta_rt = rt_a - rt_b

    print("\n" + "=" * 78)
    print("SCALING ANALYSIS & BREAK-EVEN POINT")
    print("=" * 78)

    # N* calculation
    # Time_A(N) = Prep_A + N * RT_A
    # Time_B(N) = N * (Prep_B + RT_B)
    # Break-even: Prep_A + N * RT_A = N * (Prep_B + RT_B) => N* = Prep_A / (Prep_B + RT_B - RT_A)
    denom = (prep_b + rt_b) - rt_a

    if denom > 0:
        n_break_even = math.ceil(prep_a / denom)
        print(f"\n[*] Break-even point (N*): ~{n_break_even} subregion extractions")
        print(f"    - If extracting < {n_break_even} ROIs: Strategy B (direct 2 km scenes) is faster.")
        print(f"    - If extracting >= {n_break_even} ROIs: Strategy A (10 km big scene) is faster.")
    else:
        print("\n[*] Strategy A is slower for all N because BVH Ray Tracing overhead in 10 km scene exceeds small-scene bake overhead.")

    print("\nProjections for N ROI Extractions:")
    print(f"{'N ROIs':>8} | {'Strategy A (10km Big)':>22} | {'Strategy B (2km Small)':>22} | {'Speedup Factor':>15}")
    print("-" * 75)
    for n in [1, 5, 10, 25, 50, 100, 500]:
        t_a = prep_a + (n * rt_a)
        t_b = n * (prep_b + rt_b)
        speedup = t_b / t_a if t_a > 0 else 0
        winner = "Strategy A" if t_a < t_b else "Strategy B"
        print(f"{n:>8d} | {t_a:>19.1f} s | {t_b:>19.1f} s | {speedup:>10.2f}x ({winner})")


if __name__ == "__main__":
    main()