import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Works in a headless devcontainer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from sionna import rt

ROOT = Path(__file__).resolve().parent.parent
DEEP = (ROOT / "workspace_outputs/benchmarks/deepmimo"
        / "germany_49.79130_9.95340/scene/deepmimo/scene.xml")
BAKER = (ROOT / "workspace_outputs/dataset/scenes"
         / "00000000/00000000_scenebaker_terrain.xml")
SAMPLE = ROOT / "workspace_outputs/dataset/samples/sample_00000000.npz"
OUTPUT = ROOT / "workspace_outputs/benchmarks/scene_visuals"


def footprint_polygons(mesh_dir):
    """Read DeepMIMO's ASCII PLY meshes and collect their z=0 bottom rings."""
    polygons = []
    for ply in sorted(mesh_dir.glob("building_*.ply")):
        with ply.open("r", encoding="utf-8") as file:
            vertex_count = 0
            for line in file:
                if line.startswith("element vertex "):
                    vertex_count = int(line.split()[-1])
                if line.strip() == "end_header":
                    break
            ground_points = []
            for _ in range(vertex_count):
                x, y, z = map(float, next(file).split()[:3])
                if abs(z) < 0.01:
                    ground_points.append((x, y))
            if len(ground_points) >= 3:
                polygons.append(ground_points)
    return polygons


def plot_footprints(xml, output, width):
    polygons = footprint_polygons(xml.parent / "meshes")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(PolyCollection(
        polygons, facecolors="#739dc3", edgecolors="#283f59", linewidths=0.2
    ))
    half = width / 2
    ax.set(xlim=(-half, half), ylim=(-half, half),
           xlabel="East / x (m)", ylabel="North / y (m)",
           title=f"DeepMIMO building footprints ({len(polygons):,} buildings)")
    ax.scatter([0], [0], c="red", marker="^", s=90, label="Jammer (center)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render(xml, name, ground_z, scene_width, output, samples):
    scene = rt.load_scene(str(xml), merge_shapes=True)
    # Same XY view, but compensate for absolute-vs-relative terrain altitude.
    top = rt.Camera(position=[0, -5, ground_z + scene_width * 1.6],
                    look_at=[0, 0, ground_z])
    oblique = rt.Camera(position=[-scene_width, -scene_width, ground_z + scene_width],
                        look_at=[0, 0, ground_z + 15])
    for view, camera in (("top", top), ("oblique", oblique)):
        path = output / f"{name}_{view}.png"
        scene.render_to_file(camera=camera, filename=str(path),
                             resolution=(960, 720), num_samples=samples,
                             show_devices=False)
        print(f"Saved {path}")
    del scene


def comparison(a, b, name, output):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, file, caption in zip(axes, (a, b), ("DeepMIMO · flat ground", "SceneBaker · SRTM terrain")):
        ax.imshow(plt.imread(file))
        ax.set_title(caption)
        ax.axis("off")
    fig.suptitle(f"Würzburg: identical XY area / {name} camera")
    fig.tight_layout()
    target = output / f"compare_{name}.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    print(f"Saved {target}")


def osm_quality_check(osm_file):
    """Report missing heights and complex building relations in the same OSM input."""
    if not osm_file.is_file():
        return
    root = ET.parse(osm_file).getroot()
    ways = 0
    missing_height = 0
    roof_shapes = 0
    for way in root.iter("way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if "building" not in tags:
            continue
        ways += 1
        missing_height += not any(k in tags for k in ("height", "building:height", "building:levels"))
        roof_shapes += "roof:shape" in tags
    relations = sum(
        "building" in {tag.get("k") for tag in relation.findall("tag")}
        for relation in root.iter("relation")
    )
    if ways:
        print(f"OSM building ways: {ways:,}; no explicit height/levels: "
              f"{100 * missing_height / ways:.1f}%; roof:shape tags: {roof_shapes:,}; "
              f"building relations: {relations:,}")
        print("DeepMIMO defaults missing heights to 10 m and ignores building relations/roof:shape.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmimo-xml", type=Path, default=DEEP)
    parser.add_argument("--scenebaker-xml", type=Path, default=BAKER)
    parser.add_argument("--scene-width", type=float, default=2400,
                        help="Width of generated geometry, including context (m)")
    parser.add_argument("--terrain-z", type=float,
                        help="SceneBaker ground elevation at XY origin, in metres")
    parser.add_argument("--samples", type=int, default=64,
                        help="Mitsuba rendering samples per pixel")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    deep = args.deepmimo_xml
    if not deep.is_file():
        raise FileNotFoundError(f"DeepMIMO scene missing: {deep}\nRerun with --keep-scene")

    osm_quality_check(deep.parent.parent / "map.osm")
    plot_footprints(deep, OUTPUT / "deepmimo_footprints.png", args.scene_width)
    render(deep, "deepmimo", 0.0, args.scene_width, OUTPUT, args.samples)

    baker = args.scenebaker_xml
    if not baker.is_file():
        print(f"SceneBaker scene missing: {baker}\n"
              "Set keep_scene=True in 04_generate_scene.py and rerun to compare.")
        return
    if args.terrain_z is not None:
        ground_z = args.terrain_z
    elif SAMPLE.is_file():
        with np.load(SAMPLE) as data:
            ground_z = float(data["center_elevation_m"])
    else:
        raise ValueError("Provide --terrain-z (scene center ground elevation in metres).")
    render(baker, "scenebaker", ground_z, args.scene_width, OUTPUT, args.samples)
    for view in ("top", "oblique"):
        comparison(OUTPUT / f"deepmimo_{view}.png",
                   OUTPUT / f"scenebaker_{view}.png", view, OUTPUT)
    print(f"\nVisualizations: {OUTPUT}")


if __name__ == "__main__":
    main()
