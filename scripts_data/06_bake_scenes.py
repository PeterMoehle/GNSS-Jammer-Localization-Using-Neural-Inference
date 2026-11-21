import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from sionna.rt import load_scene

# --------------------------------------------------
# 1. Paths & Imports
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

INPUT_DIR = ROOT_DIR / "workspace_inputs"
TERRAIN_CACHE_DIR = INPUT_DIR / "terrain" / "srtm"
TERRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BAKED_SCENES_DIR = INPUT_DIR / "baked_scenes"
BAKED_SCENES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Scene Specification List
# --------------------------------------------------

@dataclass
class SceneSpec:
    name: str
    lat: float
    lon: float
    range_m: float          # Total side length in meters (same for X and Y)
    osm_region: str = "germany"
    buffer_m: float = 300.0 # Extra margin so edge subregions still get external reflections


CONFIG_PATH = INPUT_DIR / "scene_configs.json"


def load_scene_specs() -> list[SceneSpec]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No scene config found at {CONFIG_PATH}. Run _visualize_scenes_to_bake.py first."
        )
    raw = json.loads(CONFIG_PATH.read_text())
    return [SceneSpec(**item) for item in raw.get("scenes", [])]


SCENES_TO_BAKE: list[SceneSpec] = load_scene_specs()
print(SCENES_TO_BAKE)


# --------------------------------------------------
# 3. Resolution & Geographic Helpers
# --------------------------------------------------

def compute_terrain_grid_size(
    total_span_m: float,
    min_subregion_m: float = 50.0,
    target_spacing_m: float = 10.0,
    min_grid: int = 128,
    max_grid: int = 1200,
) -> int:
    effective_spacing = min(target_spacing_m, min_subregion_m / 5.0)
    grid_size = int(math.ceil(total_span_m / effective_spacing))
    return max(min_grid, min(max_grid, grid_size))


def compute_bbox(lat: float, lon: float, range_m: float, buffer_m: float):
    half_span = (range_m / 2.0) + buffer_m
    dlat = half_span / 111_320.0
    dlon = half_span / (111_320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def extract_osm_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    osm_region: str,
    out_osm_file: Path,
) -> int:
    """Extracts the bounding box from pre-tiled OSM PBFs. Returns number of tiles used."""
    osm_dir = INPUT_DIR / "osm" / "prepared" / osm_region
    meta_path = osm_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared OSM metadata not found at {meta_path}. Run prepare_osm.py first."
        )

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
        cleanup_merged = False
    else:
        osm_source = out_osm_file.parent / "merged.osm.pbf"
        cleanup_merged = True
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
            "-S", "tags=building,building:part,landuse,natural,water,leisure,highway,railway",
            "--set-bounds", str(osm_source),
            "-o", str(out_osm_file),
            "--overwrite",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    if cleanup_merged and osm_source.exists():
        osm_source.unlink()

    return len(tiles)


def find_generated_xml(scene_dir: Path) -> Path:
    """Finds the Mitsuba XML file written by SceneBaker inside scene_dir."""
    xml_files = sorted(scene_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"SceneBaker did not produce an .xml file in {scene_dir}")
    for f in xml_files:
        if f.name == "scene.xml":
            return f
    return xml_files[0]


# --------------------------------------------------
# 4. Precompute & Store Walkable Map
# --------------------------------------------------

def _classify_scene_shapes(rt_scene) -> tuple[list, list, list]:
    """Classify merged Mitsuba shapes into street, park, and obstacle (building/roof/water) shapes."""
    street_shapes = []
    park_shapes = []
    obstacle_shapes = []
    ground_shapes = []

    street_tokens = ("road", "path", "pedestrian", "steps", "cycleway", "footway")
    obstacle_tokens = ("building", "roof", "wall", "water")

    for shape in rt_scene.mi_scene.shapes():
        shape_id = (shape.id() or "").lower()
        bsdf = shape.bsdf()
        bsdf_id = (bsdf.id() if bsdf is not None and bsdf.id() else "").lower()
        tag = f"{shape_id} {bsdf_id}"

        if any(tok in tag for tok in obstacle_tokens):
            obstacle_shapes.append(shape)
        elif any(tok in tag for tok in street_tokens):
            street_shapes.append(shape)
        elif "park" in tag:
            park_shapes.append(shape)
        elif "ground" in tag or "plane" in tag:
            ground_shapes.append(shape)

    if not street_shapes and not park_shapes and ground_shapes:
        street_shapes = ground_shapes

    return street_shapes, park_shapes, obstacle_shapes


def build_and_save_walkable_map(
    scene_dir: Path,
    xml_path: Path,
    usable_half_range_m: float,
    resolution_m: float = 2.0,
    ray_batch_size: int = 1_000_000,
) -> Path:
    """
    Rasterizes the full non-buffer usable scene area at `resolution_m` (default 2.0 m)
    by intersecting downward rays against the baked Mitsuba meshes, and saves:
      - x_m, y_m: 1D coordinate axes
      - height_m: float32 surface height (H, W)
      - street: bool mask of roads/paths minus buildings/water (H, W)
      - park: bool mask of parks minus buildings/water (H, W)
      - resolution_m: float scalar
    """
    t0 = time.perf_counter()
    rt_scene = load_scene(str(xml_path), merge_shapes=True)
    bbox = rt_scene.mi_scene.bbox()
    z_top = float(bbox.max.z) + 10.0

    n = int(np.floor(2.0 * usable_half_range_m / resolution_m))
    coords = -usable_half_range_m + (np.arange(n, dtype=np.float32) + 0.5) * resolution_m
    xx, yy = np.meshgrid(coords, coords)
    total_pts = xx.size

    street_shapes, park_shapes, obstacle_shapes = _classify_scene_shapes(rt_scene)
    street_ptrs = [mi.ShapePtr(s) for s in street_shapes]
    park_ptrs = [mi.ShapePtr(s) for s in park_shapes]
    obstacle_ptrs = [mi.ShapePtr(s) for s in obstacle_shapes]

    height_flat = np.full(total_pts, np.nan, dtype=np.float32)
    street_flat = np.zeros(total_pts, dtype=bool)
    park_flat = np.zeros(total_pts, dtype=bool)

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    for start in range(0, total_pts, ray_batch_size):
        end = min(start + ray_batch_size, total_pts)
        chunk_len = end - start
        origins = mi.Point3f(
            x_flat[start:end],
            y_flat[start:end],
            np.full(chunk_len, z_top, dtype=np.float32),
        )
        hits = rt_scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0, 0, -1)))
        valid = np.asarray(hits.is_valid().numpy())
        t_val = np.asarray(hits.t.numpy())
        height_flat[start:end] = np.where(valid, z_top - t_val, np.nan)

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
    np.savez_compressed(
        out_file,
        x_m=coords,
        y_m=coords,
        height_m=height_flat.reshape(n, n),
        street=street_flat.reshape(n, n),
        park=park_flat.reshape(n, n),
        resolution_m=np.float32(resolution_m),
    )

    del rt_scene
    dr.sync_thread()
    dr.drjit_malloc_trim()
    gc.collect()

    print(
        f"    Walkable map saved ({n}x{n} @ {resolution_m:g}m) "
        f"in {time.perf_counter() - t0:.2f}s -> {out_file.name}"
    )
    return out_file


# --------------------------------------------------
# 5. Bake Single Scene
# --------------------------------------------------

def _spec_matches_meta(spec: SceneSpec, meta: dict) -> bool:
    saved = meta.get("params", meta)
    return (
        saved.get("osm_region") == spec.osm_region
        and math.isclose(saved.get("lat", 0.0), spec.lat, abs_tol=1e-6)
        and math.isclose(saved.get("lon", 0.0), spec.lon, abs_tol=1e-6)
        and math.isclose(saved.get("range_m", 0.0), spec.range_m, abs_tol=1e-3)
        and math.isclose(saved.get("buffer_m", 0.0), spec.buffer_m, abs_tol=1e-3)
    )


def bake_scene(spec: SceneSpec, overwrite: bool = False, walkable_res_m: float = 2.0) -> Path:
    scene_dir = BAKED_SCENES_DIR / spec.name
    meta_file = scene_dir / "scene_meta.json"
    walkable_file = scene_dir / "walkable_map.npz"

    if meta_file.exists() and not overwrite:
        meta = json.loads(meta_file.read_text())
        xml_path = scene_dir / meta["xml_filename"]
        if xml_path.exists() and _spec_matches_meta(spec, meta):
            if not walkable_file.exists():
                print(f"[+] Building missing walkable map for '{spec.name}'...")
                build_and_save_walkable_map(
                    scene_dir, xml_path, spec.range_m / 2.0, resolution_m=walkable_res_m
                )
            else:
                print(
                    f"[✓] Skipping '{spec.name}' (already baked: "
                    f"region={spec.osm_region}, range={spec.range_m:g}m, buffer={spec.buffer_m:g}m)"
                )
            return xml_path
        elif xml_path.exists():
            print(f"[!] Parameters changed for '{spec.name}' — re-baking scene...")

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    u_lat_min, u_lat_max, u_lon_min, u_lon_max = compute_bbox(
        spec.lat, spec.lon, spec.range_m, buffer_m=0.0
    )
    lat_min, lat_max, lon_min, lon_max = compute_bbox(
        spec.lat, spec.lon, spec.range_m, spec.buffer_m
    )

    total_span_m = spec.range_m + 2.0 * spec.buffer_m
    terrain_grid_size = compute_terrain_grid_size(total_span_m)
    vertex_spacing_m = total_span_m / terrain_grid_size

    print(f"\n[+] Baking scene: '{spec.name}' [{spec.osm_region}]")
    print(f"    Center:          ({spec.lat:.6f}, {spec.lon:.6f})")
    print(f"    Usable Range:    {spec.range_m:g} m x {spec.range_m:g} m (+{spec.buffer_m:g} m buffer)")
    print(f"    Terrain Grid:    {terrain_grid_size} x {terrain_grid_size} (~{vertex_spacing_m:.1f} m spacing)")

    # 1. Extract local OSM file
    osm_file = scene_dir / "osm" / "map.osm"
    num_tiles = extract_osm_bbox(
        lat_min, lat_max, lon_min, lon_max, spec.osm_region, osm_file
    )
    t_osm = time.perf_counter()

    # 2. Bake 3D terrain + building meshes via SceneBaker
    scene = scenebaker.load_scene(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        terrain=True,
        out_dir=scene_dir,
        reuse_osm=True,
        terrain_dir=TERRAIN_CACHE_DIR,
        ground_z=0.0,
        min_ground_half_width=(spec.range_m / 2.0) + spec.buffer_m + 100.0,
        terrain_grid_size=terrain_grid_size,
    )
    t_bake = time.perf_counter()

    ground_center_z = float(scenebaker.height(scene, 0.0, 0.0, buildings=False) or 0.0)
    xml_path = find_generated_xml(scene_dir)

    del scene
    gc.collect()

    # 3. Precompute and save 2.0 m walkable map
    build_and_save_walkable_map(
        scene_dir, xml_path, spec.range_m / 2.0, resolution_m=walkable_res_m
    )
    t_walk = time.perf_counter()

    # 4. Save structured metadata including exact generation params
    metadata = {
        "name": spec.name,
        "xml_filename": xml_path.name,
        "walkable_map_filename": "walkable_map.npz",
        "walkable_resolution_m": walkable_res_m,
        "params": {
            "osm_region": spec.osm_region,
            "lat": spec.lat,
            "lon": spec.lon,
            "range_m": spec.range_m,
            "buffer_m": spec.buffer_m,
        },
        "usable_bbox": {
            "lat_min": u_lat_min,
            "lat_max": u_lat_max,
            "lon_min": u_lon_min,
            "lon_max": u_lon_max,
        },
        "buffered_bbox": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "usable_xy_half_range_m": spec.range_m / 2.0,
        "buffered_xy_half_range_m": (spec.range_m / 2.0) + spec.buffer_m,
        "center_elevation_m": ground_center_z,
        "terrain_grid_size": terrain_grid_size,
        "terrain_vertex_spacing_m": round(vertex_spacing_m, 2),
        "osm_tiles_used": num_tiles,
        "timings_s": {
            "osm_extract": round(t_osm - t0, 2),
            "scenebaker_bake": round(t_bake - t_osm, 2),
            "walkable_map": round(t_walk - t_bake, 2),
            "total": round(t_walk - t0, 2),
        },
    }
    meta_file.write_text(json.dumps(metadata, indent=2))

    print(
        f"    Done in {t_walk - t0:.2f}s "
        f"(OSM: {t_osm - t0:.2f}s, Bake: {t_bake - t_osm:.2f}s, WalkMap: {t_walk - t_bake:.2f}s) -> {xml_path.name}"
    )
    return xml_path


def update_osm_only(spec: SceneSpec) -> Path:
    scene_dir = BAKED_SCENES_DIR / spec.name
    meta_file = scene_dir / "scene_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Scene '{spec.name}' has not been baked yet.")

    lat_min, lat_max, lon_min, lon_max = compute_bbox(
        spec.lat, spec.lon, spec.range_m, spec.buffer_m
    )
    osm_file = scene_dir / "osm" / "map.osm"

    t0 = time.perf_counter()
    extract_osm_bbox(lat_min, lat_max, lon_min, lon_max, spec.osm_region, osm_file)
    print(f"[✓] OSM for '{spec.name}' updated in {time.perf_counter() - t0:.2f}s.")
    return osm_file


def update_walkable_only(spec: SceneSpec, walkable_res_m: float = 2.0) -> Path:
    scene_dir = BAKED_SCENES_DIR / spec.name
    meta_file = scene_dir / "scene_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Scene '{spec.name}' has not been baked yet.")

    meta = json.loads(meta_file.read_text())
    xml_path = scene_dir / meta["xml_filename"]
    print(f"[+] Building walkable map for '{spec.name}'...")
    out = build_and_save_walkable_map(
        scene_dir, xml_path, spec.range_m / 2.0, resolution_m=walkable_res_m
    )
    meta["walkable_map_filename"] = out.name
    meta["walkable_resolution_m"] = walkable_res_m
    meta_file.write_text(json.dumps(meta, indent=2))
    return out


# --------------------------------------------------
# 6. CLI Entrypoint
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-bake 3D city scenes for downstream Sionna RT ray tracing."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional subset of city names from SCENES_TO_BAKE to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-bake scenes even if scene_meta.json already exists.",
    )
    parser.add_argument(
        "--osm-only",
        action="store_true",
        default=False,
        help="Only re-extract map.osm for already baked scenes without re-baking 3D meshes.",
    )
    parser.add_argument(
        "--walkable-only",
        action="store_true",
        default=False,
        help="Only build/update walkable_map.npz for already baked scenes.",
    )
    parser.add_argument(
        "--walkable-res-m",
        type=float,
        default=2.0,
        help="Grid resolution in meters for the precomputed walkable map (default: 2.0).",
    )
    args = parser.parse_args()

    selected = SCENES_TO_BAKE
    if args.cities:
        wanted = {c.lower() for c in args.cities}
        selected = [s for s in SCENES_TO_BAKE if s.name.lower() in wanted]
        missing = wanted - {s.name.lower() for s in selected}
        if missing:
            raise ValueError(f"Unknown city names: {sorted(missing)}")

    print(f"Output directory: {BAKED_SCENES_DIR}")
    print(f"Scenes queued:    {[s.name for s in selected]}")

    for spec in selected:
        if args.osm_only:
            update_osm_only(spec)
        elif args.walkable_only:
            update_walkable_only(spec, walkable_res_m=args.walkable_res_m)
        else:
            bake_scene(spec, overwrite=args.overwrite, walkable_res_m=args.walkable_res_m)

    print("\nAll requested scenes are baked and ready for Sionna RT.")


if __name__ == "__main__":
    main()