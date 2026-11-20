Note: Best to first open the repo using the devcontainer as of all the sionna/dr.jit dependencies
Note: For data creation using google colab:
>import os, shutil, subprocess
>
># Ohne GPU benötigt Dr.Jit das LLVM-Backend (libLLVM)
>HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
>if not HAS_GPU and os.getenv("COLAB_RELEASE_TAG"):
>    subprocess.run("apt-get -qq update && apt-get -qq install -y llvm > /dev/null", shell=True)
>
>try:
>    import sionna.rt
>except ImportError:
>    subprocess.run("pip install -q sionna-rt==2.2.0 h5py", shell=True, check=True)
>    import sionna.rt
>
>print("GPU:", HAS_GPU, "| Sionna RT", sionna.rt.__version__)

---

## Goal: Create the data. Simpler pipeline than in 'scripts_data'. Using only predefined scenes.

### Todo:
- Currently we can sample a random scene at the beginning. But we need it for each episode. And therefore also store each individual terrain map instead of the one single. (or maybe more performant, we store the whole scene properly in the dataset and store for each entry the roi and then can compute the relevant map on the fly? --> if the data is otherwise to big in size)
  - "Add_Measurement Bugfix and rnd ROI per Scene per Episode"
- Create supernode out of wcl information (prob in dataloader) and integrate it into the node structure
- Angles that have been used in baseline paper: $[\sin\theta, \cos\theta, \sin\phi, \cos\phi]$
- Improve pipeline - using OSM-To-Sionna and street information as walkable area --> scripts_data

---

## generate_data: Creates data for a single scene. With changing sensor positions. Storing it as HDF5


### Data-Pipeline:
- load_env: Load the selected mitsuba scene; Identify walkable areas --> Note: Will be streets in future; Creates grid for the subregion of interest (ROI); 
- setup_sensors: Use farthest point sampling (or another method), to place sensors into the scene. Additionally computes the edges, combining k sensors using KNN algorithm
- generate_trajectories: Initial Jammer position is uniform selected. Facing direction as well. Propagating Jammer Position. Creating trajectory. It tries to keep up its movement direction + adding a bit of noise. Obstacles lead to change of direction. Additionally we add velocity in Jammer State Estimation 
- simulate_links: Performs Raytracing. Calculates sionna paths between jammer and sensors.
- add_measurements: Computing sensor measurements based on calculated sionna paths. Such as rssi_dbm.
- save_dataset


### Features
- **Sensoren (Knoten-Features):**
  - `rssi_dbm`: Erhaltener Signalstärkeindikator in $\text{dBm}$ (mit Option für `rssi_dbm_clamped` ohne NaN-Werte).
  - `path_gain_linear`: Linearer Pfadgewinn (Ray-Tracing-Ergebnis).
  - `path_gain_db`: Pfadgewinn in Dezibel ($\text{dB}$).
  - `num_paths`: Anzahl der reflektierten/gestreuten Signalpfade, die den Sensor erreichen.
  - `detected_mask`: Binäre Maske ($0$ oder $1$), ob das Signal über der Empfindlichkeitsschwelle (`sensitivity_dbm`) liegt.
  - `dist_to_wcl`: Euklidische Distanz des einzelnen Sensors zum berechneten WCL-Schwerpunkt im aktuellen Zeitschritt.
  - `rssi_diff_neighbor`: Lokale Signaldifferenz zwischen dem Sensor und dem Mittelwert seiner direkten Graph-Nachbarn ($\text{RSSI}_i - \bar{\text{RSSI}}_{\text{Nachbarn}}$).
  - `rssi_time_gradient`: Zeitliche Änderung der Signalstärke pro Sensor ($\Delta \text{RSSI} / \Delta t$).
- **Graph-Topologie (Kanten-Features):**
  - `dx_m`, `dy_m`, `dz_m`: Relativer 3D-Abstandsvektor zwischen verbundenen Sensoren in Metern.
  - `distance_m`: Euklidische Distanz zwischen den Sensoren in Metern.
- **Globale / Kontext-Features:**
  - `wcl_positions`: Absolute 3D-Positionen des leistungsgewichteten Schwerpunkts (Weighted Centroid Localization) pro Zeitschritt $(x, y, z)$. Dient auch als virtueller Superknoten oder Baseline.
  - `sensors/positions`: Absolute 3D-Positionen der Sensoren $(x, y, z)$ im Raum.
  - `map/free`, `map/surface_height_m`: 2D/3D-Topografie der begehbaren Umgebung.


### Targets
- `jammer/positions` $\rightarrow$ 3D-Koordinaten $(x, y, z)$ des Jammers für jeden Zeitschritt $t \in [1, T]$ und jede Episode $e$.
- `jammer/velocities` $\rightarrow$ 3D-Geschwindigkeitsvektor $(v_x, v_y, v_z)$ des Jammers pro Zeitschritt in $\text{m/s}$.
- Potential further targets to be learned:
    - `los`: Binärer Status, ob eine direkte Sichtverbindung (Line-of-Sight) zum Jammer besteht.


### Features unavailable out of simulation:
- num_paths / path_gain_linear / path_gain_db


---
## Datasets
Notes: Within the scenes ROI we should always have a bit of puffer zone for diffraction
Notes: For Multi Scene Dataset we should sample relative to city size. But capped, otherwise most of the data will just be 2 cities...
Note: Maybe we have to do the  again and include tags for highway and railway (extract_osm_bbox, so street centerlines, pedestrian zones, and plazas are included...) 
Note: Extract Building and water polygons from walkable areas so sensors dont spawn there
Note: We should build a walkable mask for a scene once and then store it as well

1. Single Scene Dataset
Single Scene - Munich / Berlin; L1 Frequency; Mixture of 3 sensor placements (farthest_spread, lhs, uniform rnd); Jammer routing via street_information; speed fixes 5m/s; SensorPlacement on Streets; ROI_Size 1 km; 5.000 episodes;
--> Most critical: Num_Sensors. Maybe better to have a range of 10-40 per scene? Should we have them fixed or at random positions?
--> Test is a set of episodes in the same city
--> Lets generate and train on 100 sensors and in test we can reduce the amount of sensors and see what is happening if we suddenly retrieve less data

2. Multi Scene Dataset

3. Faster Jammer

4. Different amount of sensors

3. Lithuania - OutOfDistribution Data



Question: Couldnt we generate within a scene 100 sensors 


---

Important Fix:
Regenerate OSM Prep and Scenebaking (but osm only).
Reason: We excluded highway information (streets...), but we need them.

Solution:
- Remove Cached prepared osm: rm -rf workspace_inputs/osm/prepared/germany
- Download the raw data via uv run sripts_data/01_sync_externals.py --> Comment out line 37 and 38, so it will download germany and lithuania
- Reexecute osm preperation: uv run sripts_data/02_prepare_osm_batched.py
- Reexecute scene baking !!! but with flag for osm only... uv run scripts/bake_scenes.py --osm-only
- Verification: grep -E 'k="(highway|railway)"' workspace_inputs/baked_scenes/<your_city_name>/osm/map.osm | head -n 10