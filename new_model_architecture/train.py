
from new_model_architecture.config import MODEL_ARCHITECTURE_DEFAULTS, TRAINING_DEFAULTS, VAL_TEST_DEFAULTS
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from pathlib import Path
import h5py

path = Path("./data/multi_scene_ds/munich_train.h5")

with h5py.File(path, "r") as file:
    print(list(file.keys()))  # sensors, jammer, measurements, graph

    sensor_positions = file["sensors/positions"][:]
    jammer_positions = file["jammer/positions"][:]
    rssi = file["measurements/rssi_dbm_clamped"][:]


print(sensor_positions.shape)
print(jammer_positions.shape)
print(rssi.shape)
def start_training_run():
    pass

if __name__ == "__main__":
    start_training_run()