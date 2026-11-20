import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

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
    """
    Computes terrain mesh vertex grid size so that:
      - Even the smallest subregion (e.g. 50 m) has >= 5 terrain vertices across.
      - Global vertex spacing stays around ~10 m (3x finer than ~30 m SRTM data),
        capped at max_grid to keep Mitsuba BVH build & memory fast.
    """
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
    # Prefer scene.xml if multiple exist
    for f in xml_files:
        if f.name == "scene.xml":
            return f
    return xml_files[0]


# --------------------------------------------------
# 4. Bake Single Scene
# --------------------------------------------------

def _spec_matches_meta(spec: SceneSpec, meta: dict) -> bool:
    """Checks if an existing baked scene was generated with the exact same parameters."""
    saved = meta.get("params", meta)  # backwards-compatible with flat meta
    return (
        saved.get("osm_region") == spec.osm_region
        and math.isclose(saved.get("lat", 0.0), spec.lat, abs_tol=1e-6)
        and math.isclose(saved.get("lon", 0.0), spec.lon, abs_tol=1e-6)
        and math.isclose(saved.get("range_m", 0.0), spec.range_m, abs_tol=1e-3)
        and math.isclose(saved.get("buffer_m", 0.0), spec.buffer_m, abs_tol=1e-3)
    )


def bake_scene(spec: SceneSpec, overwrite: bool = False) -> Path:
    scene_dir = BAKED_SCENES_DIR / spec.name
    meta_file = scene_dir / "scene_meta.json"

    if meta_file.exists() and not overwrite:
        meta = json.loads(meta_file.read_text())
        xml_path = scene_dir / meta["xml_filename"]
        if xml_path.exists() and _spec_matches_meta(spec, meta):
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

    # Usable inner bounds vs. full buffered bounds (used for OSM & terrain mesh)
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

    # 3. Save structured metadata including exact generation params
    metadata = {
        "name": spec.name,
        "xml_filename": xml_path.name,
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
            "total": round(t_bake - t0, 2),
        },
    }
    meta_file.write_text(json.dumps(metadata, indent=2))

    print(
        f"    Done in {t_bake - t0:.2f}s "
        f"(OSM: {t_osm - t0:.2f}s, Bake: {t_bake - t_osm:.2f}s) -> {xml_path.name}"
    )
    return xml_path


def update_osm_only(spec: SceneSpec) -> Path:
    scene_dir = BAKED_SCENES_DIR / spec.name
    meta_file = scene_dir / "scene_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Szene '{spec.name}' wurde noch nicht gebacken.")

    lat_min, lat_max, lon_min, lon_max = compute_bbox(
        spec.lat, spec.lon, spec.range_m, spec.buffer_m
    )
    osm_file = scene_dir / "osm" / "map.osm"
    
    t0 = time.perf_counter()
    extract_osm_bbox(lat_min, lat_max, lon_min, lon_max, spec.osm_region, osm_file)
    print(f"[✓] OSM für '{spec.name}' in {time.perf_counter() - t0:.2f}s aktualisiert.")
    return osm_file


# --------------------------------------------------
# 5. CLI Entrypoint
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
        else:
            bake_scene(spec, overwrite=args.overwrite)

    print("\nAll requested scenes are baked and ready for Sionna RT.")


if __name__ == "__main__":
    main()