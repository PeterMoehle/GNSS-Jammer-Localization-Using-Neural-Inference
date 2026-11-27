## Notes about meeting on 23.10.2026

- Students have to do a student project of 900 hours

- Poster Presentation
- Highlight the approach: How to tackle the problem
- Have we understood the problem and how to solve that?
- Demo
- 6-8 page paper

Topic: Crowdsourced GNSS-Interference Emitter Localization with Scalable Receiver Network
Satellites send signal
Receivers estimate position using the satellites

Commercial GNSS: C/N0, AGC, P

SDRs (Software-defined radios)?
- Used mainly in military, public safety, and research,
- IQ samples?? What is that?

What is jamming:
- Something that transmits Radio Frequency at the same freq as jammers
- Small jamming makes it lose precision (how much??)
- 

Question:
- How can a jammer work

- Look at these spectrograms.. damn

- How accurate of a measurement are we talking about?


Classical methods: Time based aproach
Angle of arrival
Power measurements: 

Jammer moves.

Jammer type?

Goal from Frauenhofer:
- Demonstrator: Visualization with Sionna and GUI 
- Not just flexible transmitter placement, also flexible receiver placement:
- Heatmap
- Documented obviously

What is RSSI?

Classical Benchmark: 

100m?

For each environment, error targets change depending on the thing, 

Overall Tasks
- +Look at the colab notebook and setup Sionna
- 1: Generate dataset that fits with the GNN model
- 1.5: EDA to the dataset difference between kaggle
- 2: Retrain/Fine Tune the model, compare differences
- 3: Compare or Benchmark with the "classical way"
- 3.5: Classical Methods need a lot more analysis
- 4: Think of ways to improve the accuracy

https://colab.research.google.com/drive/1bzKcDqAlGA7sCuDW_mfWWxyoMGNIVUZN?usp=sharing

## Highlight Color Meaning
- Yellow: I haven't Understand it
- Green: I didn't understand it, now I do
- Red: fun
- Blue: I need to know what this means
- Pink: what

n with long right leg is now n

## Questions on First Read
- What is inductive graph regression task?
  - Inductive: train on many graphs, test on new ones
  - Graph: Input only 1 graph
  - Regression: Output is continuous (location of jammer)

- What is a Graph Neural Network? (GNN)
  - So instead of being fully connected, a GNN layer inputs are only from the nodes it is connected to.

- Attention Based? How?
  - additional "knob" to tune how much a node listens to its inputs, *probably* multiplies with the KNN edge?
 
- What interference methods are tested here? Which ones aren't?
  - Not tested: reactive, follow-on, or barrage jammers (named in the intro but never simulated), multiple jammers, moving jammers, frequency hopping, and any real RF data. The jammer is a single static point source that’s always on.
- What is a 'multipath effect' in the interference?
- What is noise floor?

- How accurate is this model in real life signal interference?
  - That is what we need to test somehow

- Shouldn't there be signal sending or receiving somewhere?
  - In this paper, no.

- What are xwcent and dwcent?
  - they make a "centroid" based on the strength of the nodes in the surroundings
  - Weighted average of the noise and positions of the neighborhood.
  - Multiply the noise level with the position, get the average surrounding noise around itself
  - d is distance

## Problem Definition
### Definition of a Network
1. There is $N$ devices, this number can change, at least 1 device
2. The devices live in a D-dimensional Space.
3. We limit this space to a subset of the space (geographic area of interest)
4. Each device records signal measurements over time
5. Measurement contains position and "noise-floor" over time
6. Devices can either move around or stay in the same spot
7. Jammers increase this Noise Floor of nearby devices
8. Jammers is guaranteed to be in the geographic area of interest (A)
9. R is the subset of space where we do have measurements of noise
10. Jammer can also not be in R

### Measurement as Graph
1. So now we have measurements from the devices
2. Nodes are: individual measurements $(x,n)$
3. Edges are: "Spatial Relationships between measurements"
4. Edges are built by KNN, so a node is connected to each other if they are measured in a position close enough to each other
5. Edges have a weight based on the distance, if they have same position, this weight is 1
6. We then attach a feature vector to the nodes: normalized noise floor, normalized cartesian representation, normalized angular representation, and this F local
7. F local is a set of values of: median, max, diff n (how different am I from avg of neighbors?)

## Supernode:
- Extend the graph with one extra node, that listens from all nodes but don't send any values to others
- Supernode is located in the weighted average of all node positions (similar to wcent)
- 

## Assumptions of Paper (AI generated)
1. One static, always-on point jammer. The localization problem is posed around a single fixed emitter.
Interference decays with distance. This underlies the KNN locality, the $w_{ij}$ decay, and WCL itself. High noise means near the jammer.
2. WCL is good with dense, radially symmetric sampling. This is the justification for using it as a prior, taken from prior work.
3. Simulated physics is a fair stand-in. The model assumes log-distance path loss with Gaussian shadowing, a fixed −100 dBm ambient floor, and parameters drawn from fixed ranges. The paper doesn’t specify whether shadowing is drawn independently per sample, but if so it is spatially uncorrelated, which is easier than reality.
4. Test distribution equals training distribution. The classical baselines (PL, MLE, MLAT, LSQ) must estimate $\gamma$ and $P_t$ from data, while the learned models absorb the parameter distribution during training. This favors the learned models in-distribution.
5. Perfect positions, honest measurements, central fusion. There’s no GPS error, no spoofed readings, and one place collects everything.
6. Fixed coordinate frame. The model predicts absolute coordinates in a frame it has seen. Rotation augmentation hurt (58.9 vs 54.7 RMSE), consistent with this. So “extrapolating outside $R$” is learned from training placements within the same $A$, not physical extrapolation.
7. MSE in normalized 5-D space. Training is on sin/cos/r errors, while evaluation is in meters. An angular error costs more meters at larger $r$, so the loss and metric are mismatched. MSE also averages ambiguous modes, such as left versus right of the sensor cluster.

## ???
- Output is a 5-tuple of Angular Coordinates


## Things to test yourself: 
- Define Confidence-guided Adaptive Global Estimation (CAGE) and what it does

## A list of To Dos or things to try:
1. Test the paper
2. From a graph, draw how the network plays out
3. What if we use a different measurement prior other than WCL?
4. Different types of Jammers?
5. "Spatial aware Graph Calculation"? (is this calculable?)
  - Imagine that each node can "raytrace" based on its surroundings,
  - Every node knows its own blind spots, or how its own signal travels through the space
  - This is now an added prior on the map, meaning a jammer won't be there if they are in a complete blind spot?
6. What if we adapt the architecture to include multiple random trajectories?
7. How do we add the data from OpenStreetMap (OSM) to the model?
  - Does this add any more info to the model?
  - How do we generate the dataset?
8. 

"trained_model_seed1_CAGE_polar_knn3_unit_sphere_1000noise_dynamic.pth"
Seed 1, MAE: 10.231680870056152, RMSE: 15.770050849131072

"trained_model_seed1_GAT_dynamic"
Seed 1, MAE: 9.337747573852539, RMSE: 14.698194660213588

model from them
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:59<00:00, 11.53s/it]
2026-11-12 22:40:19,732 - INFO - Seed 1, MAE: 10.231679916381836, RMSE: 15.770047462599985
2026-11-12 22:40:19,757 - INFO - Saved seed 1 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 22:40:19,767 - INFO - Loading data for inference...
2026-11-12 22:40:20,273 - INFO - Loaded datasets
2026-11-12 22:40:25,499 - INFO - Creating data loaders...
2026-11-12 22:40:26,114 - INFO - Initializing model...
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:31<00:00, 10.42s/it]
2026-11-12 22:44:57,225 - INFO - Seed 2, MAE: 10.23167896270752, RMSE: 15.770049881550836
2026-11-12 22:44:57,247 - INFO - Saved seed 2 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 22:44:57,266 - INFO - Loading data for inference...
2026-11-12 22:44:57,870 - INFO - Loaded datasets
2026-11-12 22:45:05,387 - INFO - Creating data loaders...
2026-11-12 22:45:05,967 - INFO - Initializing model...
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:41<00:00, 10.83s/it]
2026-11-12 22:49:47,705 - INFO - Seed 3, MAE: 10.231678009033203, RMSE: 15.770047946390186
2026-11-12 22:49:47,741 - INFO - Saved seed 3 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 22:49:47,750 - INFO - Saved RMSE and MAE metrics to results/FirstSuccessfulTraining/metrics_statistics.csv

our trained model?
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:16<00:00,  9.85s/it]
2026-11-12 22:55:26,206 - INFO - Seed 1, MAE: 9.337745666503906, RMSE: 14.698194660213588
2026-11-12 22:55:26,221 - INFO - Saved seed 1 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 22:55:26,227 - INFO - Loading data for inference...
2026-11-12 22:55:26,720 - INFO - Loaded datasets
2026-11-12 22:55:31,968 - INFO - Creating data loaders...
2026-11-12 22:55:32,567 - INFO - Initializing model...
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:36<00:00, 10.63s/it]
2026-11-12 23:00:08,894 - INFO - Seed 2, MAE: 9.337746620178223, RMSE: 14.698194660213588
2026-11-12 23:00:08,911 - INFO - Saved seed 2 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 23:00:08,920 - INFO - Loading data for inference...
2026-11-12 23:00:09,456 - INFO - Loaded datasets
2026-11-12 23:00:15,289 - INFO - Creating data loaders...
2026-11-12 23:00:15,884 - INFO - Initializing model...
Validating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [04:37<00:00, 10.66s/it]
2026-11-12 23:04:53,040 - INFO - Seed 3, MAE: 9.337747573852539, RMSE: 14.698194660213588
2026-11-12 23:04:53,057 - INFO - Saved seed 3 test set predictions to results/FirstSuccessfulTraining/predictions_GAT_dynamic_inference.csv
2026-11-12 23:04:53,058 - INFO - Saved RMSE and MAE metrics to results/FirstSuccessfulTraining/metrics_statistics.csv


Training model...
Epochs: 50, Batch Size: 8, Learning Rate: 0.0006703373871168542, Weight Decay: 1e-05
Checkpoint saved at epoch 0 with validation loss: 114.4936                                                           
Epoch 0, Loss: 414.4735666227341
Checkpoint saved at epoch 1 with validation loss: 113.4087                                                           
Epoch 1, Loss: 95.43025741068521
Checkpoint saved at epoch 2 with validation loss: 101.3383                                                           
Epoch 2, Loss: 85.29369995244345
Epoch 3, Loss: 76.4078679842949                                                                                      
Checkpoint saved at epoch 4 with validation loss: 95.7615                                                            
Epoch 4, Loss: 67.22082656224569
Epoch 5, Loss: 63.867706633408865                                                                                    
Checkpoint saved at epoch 6 with validation loss: 88.4721                                                            
Epoch 6, Loss: 60.00264269049963
Epoch 7, Loss: 64.38678784211477                                                                                     
Epoch 8, Loss: 59.44211646842957                                                                                     
Epoch 9, Loss: 61.851821995099385                                                                                    
Epoch 10, Loss: 59.46080426311493                                                                                    
Epoch 11, Loss: 59.0778851164182                                                                                     
Checkpoint saved at epoch 12 with validation loss: 71.7773                                                           
Epoch 12, Loss: 56.27635208376249
Epoch 13, Loss: 54.927519697030384                                                                                   
Checkpoint saved at epoch 14 with validation loss: 71.5146                                                           
Epoch 14, Loss: 54.30925096893311
Epoch 15, Loss: 54.74172574679057                                                                                    
Epoch 16, Loss: 61.94714232524236                                                                                    
Epoch 17, Loss: 69.5140388468504                                                                                     
Epoch 18, Loss: 82.43391530354818                                                                                    
Epoch 19, Loss: 93.1110459596316                                                                                     
Epoch 20, Loss: 123.44733450603485                                                                                   
Epoch 21, Loss: 131.8125754289627                                                                                    
Epoch 22, Loss: 163.17620138867696                                                                                   
Epoch 23, Loss: 143.27868556722007                                                                                   
Epoch 24, Loss: 133.02604039812087                                                                                   
Epoch 25, Loss: 147.6458175983429                                                                                    
Epoch 26, Loss: 140.23278182284037                                                                                   
Epoch 27, Loss: 138.4921112124125                                                                                    
Epoch 28, Loss: 149.55021376927695                                                                                   
Epoch 29, Loss: 164.05797426064808                                                                                   
Epoch 30, Loss: 150.44479100259144                                                                                   
Epoch 31, Loss: 143.58939518260956                                                                                   
Epoch 32, Loss: 137.0643626909256                                                                                    
Epoch 33, Loss: 141.06328556760153                                                                                   
Epoch 34, Loss: 133.30718719037372                                                                                   
Epoch 35, Loss: 131.1559899530411                                                                                    
Epoch 36, Loss: 132.85230952008564                                                                                   
Epoch 37, Loss: 127.84046192916234                                                                                   
Epoch 38, Loss: 127.82769038995107                                                                                   
Epoch 39, Loss: 123.964097890536                                                                                     
Epoch 40, Loss: 119.81286337502797                                                                                   
Epoch 41, Loss: 117.69477294317882                                                                                   
Epoch 42, Loss: 114.76342136065165                                                                                   
Epoch 43, Loss: 113.40257802327474                                                                                   
Epoch 44, Loss: 109.92551623503367                                                                                   
Epoch 45, Loss: 105.40073809083303                                                                                   
Epoch 46, Loss: 102.32575177351633                                                                                   
Epoch 47, Loss: 100.38222232516607                                                                                   
Epoch 48, Loss: 97.50485810216269                                                                                    
Epoch 49, Loss: 95.0678779039383                                                                                     
Model saved to new_model_architecture/checkpoint/final_model.pth