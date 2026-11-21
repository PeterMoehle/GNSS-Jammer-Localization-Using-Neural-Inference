import json
import math
from pathlib import Path

import folium
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import st_folium

# --------------------------------------------------
# 1. Paths & Config Setup
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "workspace_inputs"
PREPARED_OSM_DIR = INPUT_DIR / "osm" / "prepared"
CONFIG_PATH = INPUT_DIR / "scene_configs.json"

# Fallback bounding boxes if prepare_osm.py hasn't been run for a region yet
KNOWN_REGIONS = {
    "germany": (5.0, 47.0, 16.0, 56.0),      # (lon_min, lat_min, lon_max, lat_max)
    "lithuania": (20.0, 53.0, 27.0, 57.0),
    "france": (-5.5, 41.0, 9.8, 51.5),
}

DEFAULT_SCENES = [
    {
        "name": "wuerzburg",
        "lat": 49.7913,
        "lon": 9.9534,
        "range_m": 10000.0,
        "buffer_m": 300.0,
        "osm_region": "germany",
    }
]


# --------------------------------------------------
# 2. Config & OSM Region Helpers
# --------------------------------------------------

def load_configs() -> list[dict]:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_configs(DEFAULT_SCENES)
        return list(DEFAULT_SCENES)
    data = json.loads(CONFIG_PATH.read_text())
    return data.get("scenes", [])


def save_configs(scenes: list[dict]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scenes": scenes}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def get_available_osm_regions() -> list[str]:
    regions = set(KNOWN_REGIONS.keys())
    if PREPARED_OSM_DIR.exists():
        for p in PREPARED_OSM_DIR.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                regions.add(p.name)
    return sorted(regions)


def detect_osm_region(lat: float, lon: float) -> str:
    """Finds which known/prepared OSM region contains (lat, lon)."""
    # First check prepared OSM directories that have meta.json
    if PREPARED_OSM_DIR.exists():
        for p in PREPARED_OSM_DIR.iterdir():
            meta_file = p / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                if "bounds" in meta:
                    lon_min, lat_min, lon_max, lat_max = meta["bounds"]
                    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                        return p.name

    for name, (lon_min, lat_min, lon_max, lat_max) in KNOWN_REGIONS.items():
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name

    return "germany"


def compute_square_bbox(lat: float, lon: float, range_m: float, buffer_m: float = 0.0):
    """Returns (lat_min, lat_max, lon_min, lon_max) for an exact metric square."""
    half_span = (range_m / 2.0) + buffer_m
    dlat = half_span / 111_320.0
    dlon = half_span / (111_320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def rect_to_centered_square(bounds_dict: dict) -> tuple[float, float, float]:
    """
    Converts a drawn leaflet rectangle into a true metric square:
    returns (center_lat, center_lon, range_m) using the larger of width/height.
    """
    sw = bounds_dict["_southWest"]
    ne = bounds_dict["_northEast"]

    center_lat = (sw["lat"] + ne["lat"]) / 2.0
    center_lon = (sw["lng"] + ne["lng"]) / 2.0

    height_m = abs(ne["lat"] - sw["lat"]) * 111_320.0
    width_m = abs(ne["lng"] - sw["lng"]) * 111_320.0 * math.cos(math.radians(center_lat))

    # Snap to nearest 50 m so numbers stay clean
    raw_side = max(width_m, height_m)
    snapped_side = max(50.0, round(raw_side / 50.0) * 50.0)
    return round(center_lat, 6), round(center_lon, 6), float(snapped_side)


# --------------------------------------------------
# 3. Streamlit App State
# --------------------------------------------------

st.set_page_config(page_title="Sionna OSM Scene Selector", layout="wide")
st.title("Sionna RT — OSM Scene & Square ROI Selector")

if "scenes" not in st.session_state:
    st.session_state.scenes = load_configs()

if "draft_name" not in st.session_state:
    st.session_state.draft_name = "new_city"
if "draft_lat" not in st.session_state:
    st.session_state.draft_lat = 49.791300
if "draft_lon" not in st.session_state:
    st.session_state.draft_lon = 9.953400
if "draft_range_m" not in st.session_state:
    st.session_state.draft_range_m = 10000.0
if "draft_buffer_m" not in st.session_state:
    st.session_state.draft_buffer_m = 300.0
if "draft_region" not in st.session_state:
    st.session_state.draft_region = detect_osm_region(49.7913, 9.9534)
if "last_draw_hash" not in st.session_state:
    st.session_state.last_draw_hash = None


# --------------------------------------------------
# 4. Layout: Left Controls | Right Interactive Map
# --------------------------------------------------

col_sidebar, col_map = st.columns([1.15, 2.3], gap="large")

with col_sidebar:
    st.subheader("1. Configure Square Scene")
    st.caption(
        "Click anywhere on the map to move the center, use the **Rectangle** draw tool "
        "on the map to sketch a square, or enter exact values below."
    )

    with st.form("scene_editor_form"):
        name = st.text_input("Scene / City Name", value=st.session_state.draft_name)

        c1, c2 = st.columns(2)
        lat = c1.number_input(
            "Center Latitude", value=float(st.session_state.draft_lat), format="%.6f", step=0.001
        )
        lon = c2.number_input(
            "Center Longitude", value=float(st.session_state.draft_lon), format="%.6f", step=0.001
        )

        c3, c4 = st.columns(2)
        range_m = c3.number_input(
            "Square Range (m)",
            min_value=50.0,
            max_value=50000.0,
            value=float(st.session_state.draft_range_m),
            step=250.0,
        )
        buffer_m = c4.number_input(
            "Buffer Margin (m)",
            min_value=0.0,
            max_value=5000.0,
            value=float(st.session_state.draft_buffer_m),
            step=50.0,
        )

        available_regions = get_available_osm_regions()
        detected = detect_osm_region(lat, lon)
        default_idx = (
            available_regions.index(st.session_state.draft_region)
            if st.session_state.draft_region in available_regions
            else 0
        )
        osm_region = st.selectbox(
            f"OSM Region (Auto-detected: {detected})",
            options=available_regions,
            index=default_idx,
        )

        lat_min, lat_max, lon_min, lon_max = compute_square_bbox(lat, lon, range_m, 0.0)
        st.code(
            f"Usable Lat: [{lat_min:.6f}, {lat_max:.6f}]\n"
            f"Usable Lon: [{lon_min:.6f}, {lon_max:.6f}]",
            language="text",
        )

        b1, b2 = st.columns(2)
        preview_btn = b1.form_submit_button("Update Preview", use_container_width=True)
        save_btn = b2.form_submit_button("Save / Upsert Scene", type="primary", use_container_width=True)

        if preview_btn or save_btn:
            clean_name = name.strip().replace(" ", "_").lower()
            st.session_state.draft_name = clean_name
            st.session_state.draft_lat = lat
            st.session_state.draft_lon = lon
            st.session_state.draft_range_m = range_m
            st.session_state.draft_buffer_m = buffer_m
            st.session_state.draft_region = osm_region

            if save_btn and clean_name:
                new_entry = {
                    "name": clean_name,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "range_m": float(range_m),
                    "buffer_m": float(buffer_m),
                    "osm_region": osm_region,
                }
                existing_idx = next(
                    (i for i, s in enumerate(st.session_state.scenes) if s["name"] == clean_name),
                    None,
                )
                if existing_idx is not None:
                    st.session_state.scenes[existing_idx] = new_entry
                else:
                    st.session_state.scenes.append(new_entry)

                save_configs(st.session_state.scenes)
                st.success(f"Saved '{clean_name}' to {CONFIG_PATH.name}")

    st.divider()
    st.subheader(f"2. Saved Scenes ({len(st.session_state.scenes)})")

    show_all = st.checkbox("Show all saved scenes on map", value=True)
    show_buffers = st.checkbox("Show outer reflection buffer rings", value=True)

    for idx, scene in enumerate(list(st.session_state.scenes)):
        with st.container(border=True):
            info_col, act_col = st.columns([3, 1.4])
            info_col.markdown(
                f"**{scene['name']}** (`{scene['osm_region']}`)  \n"
                f"`({scene['lat']:.4f}, {scene['lon']:.4f})` · "
                f"**{scene['range_m']/1000:g} km** (+{scene['buffer_m']:g} m buf)"
            )
            e_col, d_col = act_col.columns(2)
            if e_col.button("Edit", key=f"edit_{scene['name']}_{idx}", use_container_width=True):
                st.session_state.draft_name = scene["name"]
                st.session_state.draft_lat = scene["lat"]
                st.session_state.draft_lon = scene["lon"]
                st.session_state.draft_range_m = scene["range_m"]
                st.session_state.draft_buffer_m = scene["buffer_m"]
                st.session_state.draft_region = scene["osm_region"]
                st.rerun()

            if d_col.button("Del", key=f"del_{scene['name']}_{idx}", use_container_width=True):
                st.session_state.scenes.pop(idx)
                save_configs(st.session_state.scenes)
                st.rerun()

    st.download_button(
        label="Download scene_configs.json",
        data=json.dumps({"scenes": st.session_state.scenes}, indent=2),
        file_name="scene_configs.json",
        mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------
# 5. Interactive Folium Map
# --------------------------------------------------

with col_map:
    m = folium.Map(
        location=[st.session_state.draft_lat, st.session_state.draft_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (Detailed)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite Imagery",
    ).add_to(m)

    # Enable Rectangle drawing tool to sketch areas interactively
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": {"shapeOptions": {"color": "#ff6b00", "weight": 2}},
        },
        edit_options={"edit": False, "remove": False},
    ).add_to(m)

    # 1) Render all saved scenes
    if show_all:
        for s in st.session_state.scenes:
            is_editing = s["name"] == st.session_state.draft_name
            color = "#16a34a" if not is_editing else "#2563eb"

            lat_min, lat_max, lon_min, lon_max = compute_square_bbox(
                s["lat"], s["lon"], s["range_m"], 0.0
            )
            folium.Rectangle(
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                color=color,
                weight=2.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.14,
                tooltip=(
                    f"{s['name']} ({s['osm_region']}) | "
                    f"{s['range_m']/1000:g} km x {s['range_m']/1000:g} km"
                ),
            ).add_to(m)

            if show_buffers and s.get("buffer_m", 0) > 0:
                b_lat_min, b_lat_max, b_lon_min, b_lon_max = compute_square_bbox(
                    s["lat"], s["lon"], s["range_m"], s["buffer_m"]
                )
                folium.Rectangle(
                    bounds=[[b_lat_min, b_lon_min], [b_lat_max, b_lon_max]],
                    color=color,
                    weight=1.2,
                    dash_array="5, 5",
                    fill=False,
                ).add_to(m)

            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=4,
                color=color,
                fill=True,
                tooltip=f"Center: {s['name']} ({s['lat']:.5f}, {s['lon']:.5f})",
            ).add_to(m)

    # 2) Render active draft square (in bright orange)
    d_lat_min, d_lat_max, d_lon_min, d_lon_max = compute_square_bbox(
        st.session_state.draft_lat,
        st.session_state.draft_lon,
        st.session_state.draft_range_m,
        0.0,
    )
    folium.Rectangle(
        bounds=[[d_lat_min, d_lon_min], [d_lat_max, d_lon_max]],
        color="#ea580c",
        weight=3,
        fill=True,
        fill_color="#ea580c",
        fill_opacity=0.18,
        tooltip=f"ACTIVE DRAFT: {st.session_state.draft_name} ({st.session_state.draft_range_m/1000:g} km)",
    ).add_to(m)

    if show_buffers and st.session_state.draft_buffer_m > 0:
        db_lat_min, db_lat_max, db_lon_min, db_lon_max = compute_square_bbox(
            st.session_state.draft_lat,
            st.session_state.draft_lon,
            st.session_state.draft_range_m,
            st.session_state.draft_buffer_m,
        )
        folium.Rectangle(
            bounds=[[db_lat_min, db_lon_min], [db_lat_max, db_lon_max]],
            color="#ea580c",
            weight=1.5,
            dash_array="5, 5",
            fill=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    map_state = st_folium(
        m,
        height=740,
        use_container_width=True,
        key="osm_selector_map",
    )

    # Handle drawn rectangle -> convert to proper metric square
    if map_state and map_state.get("last_active_drawing"):
        drawing = map_state["last_active_drawing"]
        coords = drawing.get("geometry", {}).get("coordinates", [[]])[0]
        draw_hash = json.dumps(coords)
        if coords and draw_hash != st.session_state.last_draw_hash:
            st.session_state.last_draw_hash = draw_hash
            lons = [pt[0] for pt in coords]
            lats = [pt[1] for pt in coords]
            bounds_dict = {
                "_southWest": {"lat": min(lats), "lng": min(lons)},
                "_northEast": {"lat": max(lats), "lng": max(lons)},
            }
            c_lat, c_lon, side_m = rect_to_centered_square(bounds_dict)
            st.session_state.draft_lat = c_lat
            st.session_state.draft_lon = c_lon
            st.session_state.draft_range_m = side_m
            st.session_state.draft_region = detect_osm_region(c_lat, c_lon)
            st.rerun()

    # Handle single map click -> move center of current square
    elif map_state and map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        c_lat = round(clicked["lat"], 6)
        c_lon = round(clicked["lng"], 6)
        if (c_lat != st.session_state.draft_lat or c_lon != st.session_state.draft_lon):
            st.session_state.draft_lat = c_lat
            st.session_state.draft_lon = c_lon
            st.session_state.draft_region = detect_osm_region(c_lat, c_lon)
            st.rerun()