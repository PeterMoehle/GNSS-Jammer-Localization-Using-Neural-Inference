Goal: Create the data. Simpler pipeline than in 'scripts_data'. Using only predefined scenes.

generate_data: Creates data for a single scene. With changing sensor positions. Storing it as HDF5

Data-Pipeline:
> load_env: Load the selected mitsuba scene; Identify walkable areas --> Note: Will be streets in future; Creates grid for the subregion of interest (ROI); 
> setup_sensors: Use farthest point sampling (or another method), to place sensors into the scene. Additionally computes the edges, combining k sensors using KNN algorithm
> generate_trajectories: Initial Jammer position is uniform selected. Facing direction as well. Propagating Jammer Position. Creating trajectory. It tries to keep up its movement direction + adding a bit of noise. Obstacles lead to change of direction. Additionally we add velocity in Jammer State Estimation 
> simulate_links: Performs Raytracing. Calculates sionna paths between jammer and sensors.
> add_measurements: Computing sensor measurements based on calculated sionna paths. Such as rssi_dbm.
> save_dataset