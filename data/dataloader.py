from pathlib import Path
from typing import Sequence, Union

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


class SionnaMultiSceneDataset(Dataset):
    """PyTorch Dataset loading Sionna RT multi-scene HDF5 data.

    Yields PyTorch Geometric `Data` objects containing graph structure,
    jammer trajectory ground truth, RSSI measurements, engineered WCL & neighborhood
    features, and `detected_mask`.
    """

    def __init__(
        self,
        filepaths: Sequence[Union[str, Path]],
        rssi_key: str = "rssi_dbm_clamped",
        load_engineered_features: bool = True,
    ):
        super().__init__()
        self.filepaths = [Path(p) for p in filepaths]
        self.rssi_key = rssi_key
        self.load_engineered_features = load_engineered_features

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

        # Lazy file handles dict for multi-processing (DataLoader workers)
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

        # 3. Load Raw Measurements & Detection Mask
        rssi = torch.from_numpy(h5f[f"measurements/{self.rssi_key}"][ep_idx]).float()  # (T, N)
        detected_mask = torch.from_numpy(h5f["measurements/detected_mask"][ep_idx]).to(torch.uint8)  # (T, N)
        los = torch.from_numpy(h5f["measurements/los"][ep_idx]).bool()  # (T, N)

        # 4. Load Engineered Features (WCL & Neighborhood Dynamics)
        wcl_pos = None
        dist_to_wcl = None
        rssi_diff_neighbor = None
        rssi_time_gradient = None

        if self.load_engineered_features and "measurements/wcl_positions" in h5f:
            wcl_pos = torch.from_numpy(h5f["measurements/wcl_positions"][ep_idx]).float()  # (T, 3)
            dist_to_wcl = torch.from_numpy(h5f["measurements/dist_to_wcl"][ep_idx]).float()  # (T, N)
            rssi_diff_neighbor = torch.from_numpy(h5f["measurements/rssi_diff_neighbor"][ep_idx]).float()  # (T, N)
            rssi_time_gradient = torch.from_numpy(h5f["measurements/rssi_time_gradient"][ep_idx]).float()  # (T, N)

        # 5. Construct PyG Data Object
        data = Data(
            x=sensors_pos,                 # Base Node coordinates (N, 3)
            edge_index=edge_index,         # Graph topology (2, E_edges)
            edge_attr=edge_attr,           # Edge attributes (E_edges, 4)
            
            # Temporal Sequence Signals
            rssi=rssi,                     # Transients (T, N)
            detected_mask=detected_mask,   # Binary 0/1 Mask (T, N)
            los=los,                       # Line-of-sight status (T, N)
            
            # Engineered Features (if available)
            wcl_pos=wcl_pos,               # WCL trajectory estimate (T, 3)
            dist_to_wcl=dist_to_wcl,       # Sensor distance to WCL (T, N)
            rssi_diff_neighbor=rssi_diff_neighbor, # Local spatial signal contrast (T, N)
            rssi_time_gradient=rssi_time_gradient, # Temporal signal dynamics (T, N)
            
            # Targets
            y_pos=jammer_pos,              # Ground truth jammer positions (T, 3)
            y_vel=jammer_vel,              # Ground truth jammer velocity (T, 3)
            
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

    # Instantiate Dataset
    train_dataset = SionnaMultiSceneDataset(train_files, rssi_key="rssi_dbm_clamped")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    for batch in train_loader:
        print("\n--- PyG Batch Inspection with WCL Features ---")
        print(f"Batch object:                {batch}")
        print(f"Number of graphs in batch:   {batch.num_graphs}")
        print(f"Combined Nodes shape (x):    {batch.x.shape} -> (Batch_Size * N, 3)")
        print(f"RSSI shape:                  {batch.rssi.shape} -> (Batch_Size, T, N)")
        if hasattr(batch, "wcl_pos") and batch.wcl_pos is not None:
            print(f"WCL Positions shape:         {batch.wcl_pos.shape} -> (Batch_Size, T, 3)")
            print(f"Dist to WCL shape:           {batch.dist_to_wcl.shape} -> (Batch_Size, T, N)")
            print(f"RSSI Neighbor Diff shape:    {batch.rssi_diff_neighbor.shape} -> (Batch_Size, T, N)")
            print(f"RSSI Time Gradient shape:    {batch.rssi_time_gradient.shape} -> (Batch_Size, T, N)")
        print(f"Target Jammer Pos shape:     {batch.y_pos.shape} -> (Batch_Size, T, 3)")
        break

    # Recommended GNSS structure:
    # node_dynamic_features = torch.stack([
    #     batch.rssi,
    #     batch.dist_to_wcl,
    #     batch.rssi_diff_neighbor,
    #     batch.rssi_time_gradient
    # ], dim=-1)