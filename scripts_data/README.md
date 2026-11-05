Currently not in use yet.

Goal: Enable to load arbitrary coordinates into sionna scenes.

Why: For the case of including city environment into the model we need to 

Current State: Pipeline Exists and is ready to use.

Known Bug: Slight Cooridnate Mismatch when generating a scene.

Plan: We should later first have data on one scene, like Würzburg residence with fixed stationary sensors, then with random sensor placements. Then using more cities.

Todo: 
- 05_generate_scenes: Eventually we can precalculate bigger sionna scenes. Like whole munich
- 06_generate_data: Generate the data. Full Pipeline.
- 07_visualize: Visualize a specific data point and its measurements.

Files:
- 01_sync_externals: Retrieves external github repos (tools and baselines), as well as OpenStreetMap-Data (OSM) we want to use. Currently Germany and Lithuania.
- 02_prepare_osm: We filter the OSM-file and extract only relevant information. E.g. we dont need labels about buildings. Afterwards it creates relevant tiles. (e.g. squares with size 1km), to increase performance. Retrieving relevant OSM information out of whole germany for a specific coordinate takes much of compute. Thats why we create smaller tiles, which we then will select based on the coordinate and build together. Amount of tiles needed increases with higher ROI (Region of Interest). Tiling size should not be to small, as otherwise we have an overhead of putting them together again.
- 02_prepare_osm_batched: Same as '02_prepare_osm', just batched to be computed faster
- 03_benchmark_tiling: Tool to check computation time and analyze tiling size
- 04_generate_scene: Simple Script that generates a sionna scene from an arbitrary coordinate 