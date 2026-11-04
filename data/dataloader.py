import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Sequence, Union


class SionnaMultiSceneDataset(Dataset):
    """PyTorch Dataset loading Sionna RT multi-scene HDF5 data.
    
    Yields PyTorch Geometric `Data` objects containing graph structure,
    jammer trajectory ground truth, RSSI measurements, and `detected_mask`.
    """

    def __init__(
        self,
        filepaths: Sequence[Union[str, Path]],
        rssi_key: str = "rssi_dbm_clamped",
    ):
        super().__init__()
        self.filepaths = [Path(p) for p in filepaths]
        self.rssi_key = rssi_key

        # Build index mapping global_sample_idx -> (file_idx, episode_idx)
        self.index_map = []
        self.file_shapes = []

        for f_idx, path in enumerate(self.filepaths):
            with h5py.File(path, "r") as f:
                n_episodes = f["sensors/positions"].shape[0]
                n_steps = f["jammer/positions"].shape[1]
                n_sensors = f["sensors/positions"].shape[1]

                self.file_shapes.append((n_episodes, n_steps, n_sensors))
                for ep_idx in range(n_episodes):
                    self.index_map.append((f_idx, ep_idx))

        # Lazy file handles dict for safe multi-processing (DataLoader workers)
        self._h5_files = {}

    def _get_h5_handle(self, file_idx: int) -> h5py.File:
        """Ensure file handles are initialized per DataLoader worker process."""
        if file_idx not in self._h5_files:
            self._h5_files[file_idx] = h5py.File(self.filepaths[file_idx], "r")
        return self._h5_files[file_idx]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Data:
        file_idx, ep_idx = self.index_map[idx]
        h5f = self._get_h5_handle(file_idx)

        # 1. Load Episode Sensors & Graph Structure
        sensors_pos = torch.from_numpy(h5f["sensors/positions"][ep_idx]).float()  # (N, 3)

        # Unpack Episode Graph from packed COO format
        edge_ptr = h5f["graph/edge_ptr"]
        start_edge = edge_ptr[ep_idx]
        end_edge = edge_ptr[ep_idx + 1]

        edge_index = torch.from_numpy(
            h5f["graph/edge_index"][:, start_edge:end_edge]
        ).long()  # (2, E_edges)

        edge_attr = torch.from_numpy(
            h5f["graph/edge_attr"][start_edge:end_edge]
        ).float()  # (E_edges, 4) -> [dx, dy, dz, distance]

        # 2. Load Trajectory Ground Truth
        jammer_pos = torch.from_numpy(h5f["jammer/positions"][ep_idx]).float()  # (T, 3)
        jammer_vel = torch.from_numpy(h5f["jammer/velocities"][ep_idx]).float()  # (T, 3)

        # 3. Load Measurements & Detection Mask
        rssi = torch.from_numpy(h5f[f"measurements/{self.rssi_key}"][ep_idx]).float()  # (T, N)
        detected_mask = torch.from_numpy(h5f["measurements/detected_mask"][ep_idx]).uint8()  # (T, N)
        los = torch.from_numpy(h5f["measurements/los"][ep_idx]).bool()  # (T, N)

        # 4. Construct PyG Data Object
        data = Data(
            x=sensors_pos,                 # Node coordinates (N, 3)
            edge_index=edge_index,         # Graph topology (2, E_edges)
            edge_attr=edge_attr,           # Sensor distance/orientation features (E_edges, 4)
            
            # Temporal Sequence Signals & Target Attributes
            rssi=rssi,                     # Transients (T, N)
            detected_mask=detected_mask,   # Binary 0/1 Mask (T, N)
            los=los,                       # Line-of-sight status (T, N)
            
            y_pos=jammer_pos,              # Target Jammer Trajectory Positions (T, 3)
            y_vel=jammer_vel,              # Target Jammer Trajectory Velocity (T, 3)
            
            episode_idx=ep_idx,
            scene_name=h5f.attrs.get("scene_name", "unknown")
        )
        return data

    def __del__(self):
        """Close open HDF5 file handles upon garbage collection."""
        for hf in self._h5_files.values():
            try:
                hf.close()
            except Exception:
                pass


# ==============================================================================
# Usage Example & Batching Verification
# ==============================================================================
if __name__ == "__main__":
    train_files = ["data/sionna_multi_scene/munich_train.h5", "data/sionna_multi_scene/florence_train.h5"]
    test_files = ["data/sionna_multi_scene/etoile_test.h5"]

    # Instantiate Datasets
    train_dataset = SionnaMultiSceneDataset(train_files, rssi_key="rssi_dbm_clamped")
    test_dataset = SionnaMultiSceneDataset(test_files, rssi_key="rssi_dbm_clamped")

    print(f"Total Training Episodes across scenes: {len(train_dataset)}")
    print(f"Total Test Episodes in held-out scene: {len(test_dataset)}")

    # Create PyG DataLoaders (Handles dynamic graph batching automatically)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    # Inspect one batch from the DataLoader
    for batch in train_loader:
        print("\n--- PyG Batch Inspection ---")
        print(f"Batch object:               {batch}")
        print(f"Number of graphs in batch:  {batch.num_graphs}")
        print(f"Combined Nodes shape (x):   {batch.x.shape} -> (Batch_Size * N, 3)")
        print(f"Edge Index shape:           {batch.edge_index.shape}")
        print(f"RSSI shape:                 {batch.rssi.shape} -> (Batch_Size, T, N)")
        print(f"Detected Mask shape:        {batch.detected_mask.shape} -> (Batch_Size, T, N)")
        print(f"Target Jammer Pos shape:    {batch.y_pos.shape} -> (Batch_Size, T, 3)")
        print(f"Overall Batch Detection %: {batch.detected_mask.float().mean() * 100:.2f}%")
        break