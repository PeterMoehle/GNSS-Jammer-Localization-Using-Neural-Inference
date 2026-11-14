import gc
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sionna import rt

# --------------------------------------------------
# 1. Setup
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
DATASET_DIR = ROOT_DIR / "workspace_outputs" / "dataset"

TERRAIN_CACHE_DIR = INPUT_DIR / "terrain" / "srtm"
TERRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = DATASET_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Scene
# --------------------------------------------------

sample_id = 0

lat = 49.792996
lon = 9.937930

osm_region = "germany"

scene_size_m = 2_000
range_m = scene_size_m / 2

buffer_m = 200
scene_range_m = range_m + buffer_m

grid_res = 96

tx_height = 3.0
rx_height = 1.5
tx_power_dbm = 35

keep_scene = True


# --------------------------------------------------
# 3. Geographic bbox
# --------------------------------------------------

dlat = scene_range_m / 111_320
dlon = scene_range_m / (111_320 * np.cos(np.radians(lat)))

lat_min = lat - dlat
lat_max = lat + dlat
lon_min = lon - dlon
lon_max = lon + dlon


# --------------------------------------------------
# 4. Prepared OSM tiles
# --------------------------------------------------

osm_dir = INPUT_DIR / "osm" / "prepared" / osm_region
meta = json.loads((osm_dir / "meta.json").read_text())

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


# --------------------------------------------------
# 5. Scene directory
# --------------------------------------------------

scene_dir = DATASET_DIR / "scenes" / f"{sample_id:08d}"

if scene_dir.exists():
    shutil.rmtree(scene_dir)

osm_file = scene_dir / "osm" / "map.osm"
osm_file.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 6. Merge relevant tiles + exact bbox
# --------------------------------------------------

t0 = time.perf_counter()

if len(tiles) == 1:
    osm_source = tiles[0]

else:
    osm_source = scene_dir / "merged.osm.pbf"

    subprocess.run(
        [
            "osmium",
            "merge",
            *map(str, tiles),
            "-o", str(osm_source),
            "--overwrite",
        ],
        check=True,
    )

subprocess.run(
    [
        "osmium",
        "extract",
        "-b", f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "-s", "smart",
        "-S", "tags=building,building:part,landuse,natural,water,leisure",
        "--set-bounds",
        str(osm_source),
        "-o", str(osm_file),
        "--overwrite",
    ],
    check=True,
)

t_osm = time.perf_counter()


# """""""""""""""""
import xml.etree.ElementTree as ET

bounds = ET.parse(osm_file).getroot().find("bounds")

actual_lat = (
    float(bounds.get("minlat")) + float(bounds.get("maxlat"))
) / 2

actual_lon = (
    float(bounds.get("minlon")) + float(bounds.get("maxlon"))
) / 2

dx = (actual_lon - lon) * 111_320 * np.cos(np.radians(lat))
dy = (actual_lat - lat) * 111_320

print(f"Requested center: {lat:.8f}, {lon:.8f}")
print(f"Scene origin:     {actual_lat:.8f}, {actual_lon:.8f}")
print(f"Offset:           east={dx:.3f} m, north={dy:.3f} m")

assert np.hypot(dx, dy) < 0.1, "OSM scene origin does not match requested GPS!"


# --------------------------------------------------
# 7. SceneBaker
#
# reuse_osm=True -> no Overpass.
#
# ground_z=0.0 means the terrain mesh sits directly at the
# SRTM elevation instead of SceneBaker's default -0.8 m offset.
# --------------------------------------------------

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
    min_ground_half_width=scene_range_m + 100,
    terrain_grid_size=grid_res,
)

t_scene = time.perf_counter()


# --------------------------------------------------
# 8. Terrain + buildings
# --------------------------------------------------

ground_center = scenebaker.height(scene, 0, 0, buildings=False)

if ground_center is None:
    raise RuntimeError("Could not determine center ground elevation.")

size = 2 * range_m
cell_size = size / grid_res

# Radio-map cell centers
coords = (
    np.arange(grid_res) + 0.5
) * cell_size - range_m

world = np.zeros((2, grid_res, grid_res), dtype=np.float32)

for iy, y in enumerate(coords):
    for ix, x in enumerate(coords):
        ground = scenebaker.height(scene, x, y, buildings=False)
        top = scenebaker.height(scene, x, y, buildings=True)

        if ground is None:
            continue

        if top is None:
            top = ground

        world[0, iy, ix] = ground - ground_center
        world[1, iy, ix] = max(0, top - ground)

t_world = time.perf_counter()


# --------------------------------------------------
# 9. Sionna RT
# --------------------------------------------------

scene.frequency = 1.57542e9

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

scene.add(
    rt.Transmitter(
        name="jammer",
        position=[0, 0, ground_center + tx_height],
    )
)


# --------------------------------------------------
# 10. Radio map
#
# This is currently a flat receiver plane.
# --------------------------------------------------

radio_map = rt.RadioMapSolver()(
    scene,
    center=[0, 0, ground_center + rx_height],
    orientation=[0, 0, 0],
    size=[size, size],
    cell_size=(cell_size, cell_size),
    max_depth=3,
    samples_per_tx=50_000,
)

gain = np.asarray(radio_map.path_gain)[0]
gain_db = 10 * np.log10(np.maximum(gain, 1e-14))

coverage_dbm = tx_power_dbm + gain_db

t_radio = time.perf_counter()


# --------------------------------------------------
# 11. Save
# --------------------------------------------------

output_file = SAMPLES_DIR / f"sample_{sample_id:08d}.npz"

np.savez_compressed(
    output_file,
    world=world,
    coverage_dbm=coverage_dbm,
    lat=lat,
    lon=lon,
    center_elevation_m=ground_center,
    scene_size_m=scene_size_m,
    grid_res=grid_res,
    tx_height_m=tx_height,
    rx_height_m=rx_height,
    tx_power_dbm=tx_power_dbm,
)

t_save = time.perf_counter()


# --------------------------------------------------
# 12. Summary
# --------------------------------------------------

print()
print(f"Scene:      {lat:.6f}, {lon:.6f}")
print(f"OSM region: {osm_region}")
print(f"OSM tiles:  {len(tiles)} x {tile_deg:g}°")
print(f"Elevation:  {ground_center:.1f} m")
print(f"Terrain:    {world[0].min():.1f} ... {world[0].max():.1f} m")
print(f"Buildings:  max {world[1].max():.1f} m")
print(f"Coverage:   {coverage_dbm.shape}")
print(f"Saved:      {output_file}")

print()
print("Timing")
print(f"  OSM:        {t_osm - t0:.2f} s")
print(f"  SceneBaker: {t_scene - t_osm:.2f} s")
print(f"  World:      {t_world - t_scene:.2f} s")
print(f"  Sionna:     {t_radio - t_world:.2f} s")
print(f"  Save:       {t_save - t_radio:.2f} s")
print(f"  Total:      {t_save - t0:.2f} s")


# --------------------------------------------------
# 12. Pretty render
# --------------------------------------------------

# --------------------------------------------------
# 13. Visualization
# --------------------------------------------------

from sionna.rt import Camera

print("Rendering scene...")

# Lower oblique angle to reveal terrain relief
camera = Camera(
    position=[-1300, -1100, ground_center + 280],
    look_at=[0, 0, ground_center + 20],
)

render_file = scene_dir / "scene_pretty.png"

scene.render_to_file(
    camera=camera,
    filename=str(render_file),
    resolution=(1280, 720),
    num_samples=64,
    fov=60,
    lighting_scale=1.5,
)

print(f"Saved: {render_file}")

import open3d as o3d

terrain_ply = scene_dir / "meshes_python_terrain" / "Plane.ply"

terrain_mesh = o3d.io.read_triangle_mesh(str(terrain_ply))
terrain_z = np.asarray(terrain_mesh.vertices)[:, 2]

print(f"Terrain mesh vertices: {len(terrain_z)}")
print(f"Terrain mesh min Z:    {terrain_z.min():.2f} m")
print(f"Terrain mesh max Z:    {terrain_z.max():.2f} m")
print(f"Terrain mesh relief:   {np.ptp(terrain_z):.2f} m")

assert np.ptp(terrain_z) > 1.0, "Terrain mesh appears flat!"

top_camera = Camera(
    position=[0, 0, ground_center + 2500],
    look_at=[0, 0, ground_center],
)

scene.render_to_file(
    camera=top_camera,
    filename=str(scene_dir / "scene_top.png"),
    resolution=(1024, 1024),
    num_samples=32,
)
# --------------------------------------------------
# 13. Cleanup
# --------------------------------------------------

if not keep_scene:
    del scene
    gc.collect()
    shutil.rmtree(scene_dir, ignore_errors=True)