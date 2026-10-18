import argparse
import math
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OSM_DIR = ROOT_DIR / "workspace_inputs" / "osm" / "prepared" / "germany"

LOCATIONS = [
    ("Würzburg", 49.7913, 9.9534),
    ("Berlin", 52.5200, 13.4050),
    ("Hamburg", 53.5511, 9.9937),
    ("Munich", 48.1351, 11.5820),
    ("Cologne", 50.9375, 6.9603),
    ("Frankfurt", 50.1109, 8.6821),
    ("Stuttgart", 48.7758, 9.1829),
    ("Leipzig", 51.3397, 12.3731),
    ("Dresden", 51.0504, 13.7373),
    ("Hannover", 52.3759, 9.7320),
    ("Tile boundary", 50.0, 10.0),
]


# --------------------------------------------------
# Geographic helpers
# --------------------------------------------------

def bbox(lat, lon, scene_size, buffer):
    radius = scene_size / 2 + buffer
    dlat = radius / 111_320
    dlon = radius / (111_320 * math.cos(math.radians(lat)))

    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def relevant_tiles(bounds, degree):
    west, south, east, north = bounds
    directory = OSM_DIR / f"tiles_{degree:g}".replace(".", "p")

    tiles = []

    for iy in range(
        math.floor(south / degree + 1e-9),
        math.floor(north / degree - 1e-9) + 1,
    ):
        for ix in range(
            math.floor(west / degree + 1e-9),
            math.floor(east / degree - 1e-9) + 1,
        ):
            path = directory / f"{iy}_{ix}.osm.pbf"

            if not path.is_file():
                raise FileNotFoundError(f"Missing tile: {path}")

            tiles.append(path)

    return tiles


# --------------------------------------------------
# Benchmark one scene
# --------------------------------------------------

def benchmark_one(name, lat, lon, degree, scene_size, buffer):
    bounds = bbox(lat, lon, scene_size, buffer)
    tiles = relevant_tiles(bounds, degree)

    input_mb = sum(p.stat().st_size for p in tiles) / 1024**2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        start = time.perf_counter()

        if len(tiles) == 1:
            source = tiles[0]
        else:
            source = tmp / "merged.osm.pbf"

            subprocess.run(
                ["osmium", "merge", *map(str, tiles), "-o", str(source)],
                check=True,
                stdout=subprocess.DEVNULL,
            )

        merge_time = time.perf_counter() - start

        output = tmp / "map.osm"

        subprocess.run(
            [
                "osmium", "extract",
                "-b", ",".join(map(str, bounds)),
                "-s", "smart",
                "-S", "tags=building,building:part,landuse,natural,water,leisure",
                "--set-bounds",
                str(source),
                "-o", str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        total_time = time.perf_counter() - start
        output_mb = output.stat().st_size / 1024**2

    print(
        f"{name:16s} "
        f"{total_time:7.2f}s  "
        f"tiles={len(tiles):2d}  "
        f"input={input_mb:7.1f} MB  "
        f"merge={merge_time:6.2f}s  "
        f"output={output_mb:6.1f} MB"
    )

    return total_time, len(tiles), input_mb


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene-size", type=float, default=2_000)
    parser.add_argument("--buffer", type=float, default=200)
    parser.add_argument("--sizes", nargs="+", type=float,
                        default=[0.1, 0.5, 1.0, 4.0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=100_000)

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    print(f"OSM:       {OSM_DIR}")
    print(f"Scene:     {args.scene_size:g} m")
    print(f"Buffer:    {args.buffer:g} m")
    print(f"Locations: {len(LOCATIONS)}")
    print()

    results = []

    for degree in args.sizes:
        directory = OSM_DIR / f"tiles_{degree:g}".replace(".", "p")

        if not directory.is_dir():
            raise FileNotFoundError(f"Tile directory missing: {directory}")

        print(f"\n{'=' * 65}")
        print(f"Tile size: {degree:g}°")
        print(f"{'=' * 65}")

        timings = []
        tile_counts = []
        input_sizes = []

        for _ in range(args.repeats):
            for name, lat, lon in LOCATIONS:
                elapsed, count, size = benchmark_one(
                    name, lat, lon, degree,
                    args.scene_size, args.buffer,
                )
                timings.append(elapsed)
                tile_counts.append(count)
                input_sizes.append(size)

        mean = statistics.mean(timings)
        median = statistics.median(timings)
        avg_tiles = statistics.mean(tile_counts)
        avg_input = statistics.mean(input_sizes)

        results.append((degree, mean, median, avg_tiles, avg_input))

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"{'Degree':>8} {'Mean':>10} {'Median':>10} {'Tiles':>10} {'Input MB':>12}")

    for degree, mean, median, tiles, input_mb in results:
        print(
            f"{degree:>8g} "
            f"{mean:>9.2f}s "
            f"{median:>9.2f}s "
            f"{tiles:>10.2f} "
            f"{input_mb:>12.1f}"
        )

    best = min(results, key=lambda r: r[1])

    print(f"\nFastest existing tile size: {best[0]:g}°")
    print(f"Mean extraction:           {best[1]:.2f} s")
    print(
        f"{args.dataset_size:,} scenes:             "
        f"{best[1] * args.dataset_size / 3600:.1f} serial extraction-hours"
    )


if __name__ == "__main__":
    main()