from pathlib import Path
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# External repositories
# --------------------------------------------------
REPOS = {
    ROOT_DIR / "externals" / "baseline":
        "https://github.com/tiiuae/gnn-jamming-source-localization.git",

    ROOT_DIR / "externals" / "tools" / "sionna-rt-gui":
        "https://github.com/NVlabs/sionna-rt-gui.git",

    ROOT_DIR / "externals" / "tools" / "sionna-scene-baker":
        "https://github.com/hslyu/sionna-scene-baker.git",

    ROOT_DIR / "externals" / "tools" / "sionna-large-radio-maps":
        "https://github.com/NVlabs/sionna-large-radio-maps.git",

    ROOT_DIR / "externals" / "tools" / "geo2sigmap":
        "https://github.com/functions-lab/geo2sigmap.git",

    ROOT_DIR / "externals" / "tools" / "SionnaRTStudio":
        "https://github.com/puloktarafder/SionnaRTStudio.git",
}


# --------------------------------------------------
# Raw OSM data
# --------------------------------------------------
OSM_DIR = ROOT_DIR / "workspace_inputs" / "osm" / "raw"

OSM_DOWNLOADS = {
#    "germany": "https://download.geofabrik.de/europe/germany-latest.osm.pbf",
#    "france": "https://download.geofabrik.de/europe/france-latest.osm.pbf", If we want to include triumph de arc
#    "lithuania": "https://download.geofabrik.de/europe/lithuania-latest.osm.pbf",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clone_or_update(target: Path, url: str):
    target.parent.mkdir(parents=True, exist_ok=True)

    if not (target / ".git").exists():
        print(f"[+] Cloning {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    else:
        print(f"[✓] Updating {target.relative_to(ROOT_DIR)}")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)


def download_osm(name: str, url: str):
    OSM_DIR.mkdir(parents=True, exist_ok=True)

    target = OSM_DIR / f"{name}.osm.pbf"
    partial = target.with_suffix(".osm.pbf.part")

    if target.exists():
        print(f"[✓] OSM exists: {target.relative_to(ROOT_DIR)}")
        return

    print(f"[+] Downloading {name} OSM")

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry", "3",
            "--continue-at", "-",
            "-o", str(partial),
            url,
        ],
        check=True,
    )

    partial.rename(target)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    for path, url in REPOS.items():
        clone_or_update(path, url)

    # Install geo2sigmap's scene_generation subpackage into the active uv venv (editable, no-deps to keep lock clean)
    geo2sigmap_pkg = ROOT_DIR / "externals" / "tools" / "geo2sigmap" / "package"
    if geo2sigmap_pkg.exists():
        print("[+] Linking geo2sigmap (scene_generation) into uv environment...")
        subprocess.run(["uv", "pip", "install", "--no-deps", "-e", str(geo2sigmap_pkg)], check=False)

    editable_packages = [
        ROOT_DIR / "externals" / "tools" / "geo2sigmap" / "package",
        ROOT_DIR / "externals" / "tools" / "sionna-rt-gui",
        ROOT_DIR / "externals" / "tools" / "sionna-scene-baker",
    ]

    for path in editable_packages:
        if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
            print(f"[+] Linking {path.name}")
            subprocess.run(["uv", "pip", "install", "--no-deps", "-e", str(path)], check=True)

    for name, url in OSM_DOWNLOADS.items():
        download_osm(name, url)

    print("\nDone.")


if __name__ == "__main__":
    main()