Note: Best to first open the repo using the devcontainer as of all the sionna/dr.jit dependencies

---

## Goal: Create the data. Simpler pipeline than in 'scripts_data'. Using only predefined scenes.

### Todo:
- Create supernode out of wcl information (prob in dataloader) and integrate it into the node structure
- Angles that have been used in baseline paper: $[\sin\theta, \cos\theta, \sin\phi, \cos\phi]$
- Improve pipeline - using OSM-To-Sionna and street information as walkable area --> scripts_data

---

## generate_data: Creates data for a single scene. With changing sensor positions. Storing it as HDF5


### Data-Pipeline:
> load_env: Load the selected mitsuba scene; Identify walkable areas --> Note: Will be streets in future; Creates grid for the subregion of interest (ROI); 
> setup_sensors: Use farthest point sampling (or another method), to place sensors into the scene. Additionally computes the edges, combining k sensors using KNN algorithm
> generate_trajectories: Initial Jammer position is uniform selected. Facing direction as well. Propagating Jammer Position. Creating trajectory. It tries to keep up its movement direction + adding a bit of noise. Obstacles lead to change of direction. Additionally we add velocity in Jammer State Estimation 
> simulate_links: Performs Raytracing. Calculates sionna paths between jammer and sensors.
> add_measurements: Computing sensor measurements based on calculated sionna paths. Such as rssi_dbm.
> save_dataset


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
