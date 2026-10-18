# uv run scripts/prepare_osm.py --region germany --scene-size 2000 --buffer 200
# for a 2km scene + 200m buffer --> should choose 0.1°
import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "workspace_inputs" / "osm" / "raw"
PREPARED_DIR = ROOT_DIR / "workspace_inputs" / "osm" / "prepared"


# --------------------------------------------------
# Regions
# --------------------------------------------------
REGIONS = {
    "germany": {
        "bounds": (5.0, 47.0, 16.0, 56.0),
        "reference_lat": 51.0,
    },
    "lithuania": {
        "bounds": (20.0, 53.0, 27.0, 57.0),
        "reference_lat": 55.0,
    },
}


# --------------------------------------------------
# SceneBaker-relevant OSM tags
# --------------------------------------------------
FILTERS = [
    "a/building",
    "a/building:part",

    "w/highway=primary,primary_link,trunk,trunk_link,"
    "secondary,secondary_link,tertiary,tertiary_link,"
    "residential,living_street,service,unclassified,"
    "track,rest_area,construction,pedestrian,footway,"
    "bridleway,cycleway,steps,path",

    "a/natural=water,wood,tree,tree_row,shrubbery,heath,grassland,scrub",
    "a/water",
    "a/landuse=forest,grass,shrubs,flowerbed",
    "a/leisure=park,garden,pitch",
]

SMART_TAGS = "building,building:part,landuse,natural,water,leisure"


# --------------------------------------------------
# Tile-size selection
# --------------------------------------------------
TILE_CHOICES = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]


def choose_tile_deg(scene_size_m: float, buffer_m: float, latitude: float) -> float:
    context_size_m = scene_size_m + 2 * buffer_m

    # Aim for a tile roughly 3x the complete context scene width.
    target_tile_m = 3 * context_size_m

    meters_per_lon_degree = 111_320 * math.cos(math.radians(latitude))
    desired_deg = target_tile_m / meters_per_lon_degree

    return min(TILE_CHOICES, key=lambda d: abs(math.log(d / desired_deg)))


def tile_levels(final_deg: float):
    if final_deg == 1.0: return [4.0, 1.0]
    if final_deg == 0.5: return [4.0, 1.0, 0.5]
    if final_deg == 0.25: return [4.0, 1.0, 0.5, 0.25]
    if final_deg == 0.1: return [4.0, 1.0, 0.5, 0.1]
    if final_deg == 0.05: return [4.0, 1.0, 0.5, 0.1, 0.05]
    if final_deg == 0.025: return [4.0, 1.0, 0.5, 0.1, 0.05, 0.025]

    raise ValueError(final_deg)


# --------------------------------------------------
# Tile helpers
# --------------------------------------------------
def tile_label(deg: float):
    return f"{deg:g}".replace(".", "p")


def tile_name(lat_i: int, lon_i: int):
    return f"{lat_i}_{lon_i}.osm.pbf"


def tile_indices(bounds, degree):
    lon_min, lat_min, lon_max, lat_max = bounds
    eps = 1e-10

    lon_start = math.floor(lon_min / degree)
    lon_end = math.floor((lon_max - eps) / degree)

    lat_start = math.floor(lat_min / degree)
    lat_end = math.floor((lat_max - eps) / degree)

    for lat_i in range(lat_start, lat_end + 1):
        for lon_i in range(lon_start, lon_end + 1):
            yield lat_i, lon_i


def valid(path: Path):
    return path.exists() and path.stat().st_size > 0


# --------------------------------------------------
# OSM commands
# --------------------------------------------------
def filter_osm(source: Path, output: Path):
    if valid(output):
        print(f"[✓] Filtered OSM exists: {output.name}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    print("[+] Filtering SceneBaker-relevant OSM features")

    subprocess.run(
        [
            "osmium",
            "tags-filter",
            str(source),
            *FILTERS,
            "-o", str(output),
            "--overwrite",
        ],
        check=True,
    )


def extract(source: Path, output: Path, bbox):
    if valid(output):
        return

    if output.exists():
        output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox

    subprocess.run(
        [
            "osmium",
            "extract",
            "-b", f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "-s", "smart",
            "-S", f"tags={SMART_TAGS}",
            "--set-bounds",
            str(source),
            "-o", str(output),
            "--overwrite",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


# --------------------------------------------------
# Build one tile level
# --------------------------------------------------
def build_level(region_dir: Path, bounds, degree: float, parent_degree: float | None, source: Path):
    output_dir = region_dir / f"tiles_{tile_label(degree)}"
    complete_file = output_dir / ".complete"

    if complete_file.exists():
        print(f"[✓] {degree:g}° tiles already complete")
        return output_dir

    indices = list(tile_indices(bounds, degree))

    print(f"[+] Building {len(indices)} x {degree:g}° tiles")

    parent_dir = None
    if parent_degree is not None:
        parent_dir = region_dir / f"tiles_{tile_label(parent_degree)}"

    for n, (lat_i, lon_i) in enumerate(indices, start=1):
        output = output_dir / tile_name(lat_i, lon_i)

        if valid(output):
            continue

        lat0 = lat_i * degree
        lon0 = lon_i * degree

        tile_source = source

        if parent_dir is not None:
            parent_lat_i = math.floor((lat0 + 1e-9) / parent_degree)
            parent_lon_i = math.floor((lon0 + 1e-9) / parent_degree)

            tile_source = parent_dir / tile_name(parent_lat_i, parent_lon_i)

            if not valid(tile_source):
                raise FileNotFoundError(f"Missing parent tile: {tile_source}")

        print(f"\r    {n}/{len(indices)}", end="", flush=True)

        extract(tile_source, output, (lon0, lat0, lon0 + degree, lat0 + degree))

    print()

    complete_file.touch()
    return output_dir


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--region", choices=REGIONS, default="germany")
    parser.add_argument("--scene-size", type=float, default=2_000)
    parser.add_argument("--buffer", type=float, default=200)

    args = parser.parse_args()

    region = REGIONS[args.region]

    raw_file = RAW_DIR / f"{args.region}.osm.pbf"
    region_dir = PREPARED_DIR / args.region
    filtered_file = region_dir / "filtered.osm.pbf"

    if not raw_file.exists():
        raise FileNotFoundError(f"Missing {raw_file}. Run sync_externals.py first.")

    tile_deg = choose_tile_deg(
        args.scene_size,
        args.buffer,
        region["reference_lat"],
    )

    context_size = args.scene_size + 2 * args.buffer
    tile_width_km = tile_deg * 111.32 * math.cos(math.radians(region["reference_lat"]))

    print(f"Region:       {args.region}")
    print(f"Scene:        {args.scene_size / 1000:.1f} km")
    print(f"Context:      {context_size / 1000:.1f} km")
    print(f"Tile size:    {tile_deg:g}° (~{tile_width_km:.1f} km east-west)")
    print()

    filter_osm(raw_file, filtered_file)

    levels = tile_levels(tile_deg)

    previous = None

    for degree in levels:
        build_level(
            region_dir=region_dir,
            bounds=region["bounds"],
            degree=degree,
            parent_degree=previous,
            source=filtered_file,
        )

        previous = degree

    metadata = {
        "region": args.region,
        "scene_size_m": args.scene_size,
        "buffer_m": args.buffer,
        "tile_deg": tile_deg,
        "tile_dir": f"tiles_{tile_label(tile_deg)}",
    }

    (region_dir / "meta.json").write_text(json.dumps(metadata, indent=2))

    print()
    print(f"Prepared OSM: {region_dir}")
    print(f"Final tiles:  {region_dir / metadata['tile_dir']}")


if __name__ == "__main__":
    main()