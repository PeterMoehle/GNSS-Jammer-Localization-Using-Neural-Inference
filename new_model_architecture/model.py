from torch._tensor import Tensor

from torch._tensor import Tensor

from torch.nn.modules.dropout import Dropout

from torch.nn.modules.linear import Linear

from torch_geometric.nn.aggr.attention import AttentionalAggregation


import warnings
from types import NoneType

import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import (
    GAT,
    GCN,
    MLP,
    PNA,
    AttentionalAggregation,
    global_max_pool,
)
from typing import Any

warnings.filterwarnings('ignore', category=UserWarning)

from config import MODEL_ARCHITECTURE_DEFAULTS, TRAINING_DEFAULTS, VAL_TEST_DEFAULTS

class GNN(torch.nn.Module):
    """
    GNN class to predict jammer coordinates based on network data.

    This GNN model can be configured to use various types of graph convolutional layers, including
    Multi-Layer Perceptrons (MLP), Graph Convolutional Network (GCN), Graph Attention Network (GAT),
    and Principal Neighbors Aggregation (PNA). The model architecture includes configurable dropout
    and fully connected layers for regression of the target features.

    Args:
        in_channels (int): Number of input features per node.
        dropout_rate (float): Dropout rate used for regularization to prevent overfitting.
        num_heads (int): Number of attention heads in GAT layers, used only if GAT is selected as model_type.
        model_type (str): Type of GNN to use ('MLP', 'GCN', 'GAT', 'GATv2', 'PNA').
        hidden_channels (int): Number of hidden units in each GNN layer.
        out_channels (int): Number of output features from the last GNN layer before regression.
        num_layers (int): Number of layers in the GNN.
        act (str, optional): Activation function to use (default 'relu').
        norm (str, optional): Type of normalization layer to use, if any (default None).
        deg (np.ndarray, optional): Degree information for nodes, used only in PNA for scalable aggregations (default None).

    Attributes:
        gnn (torch.nn.Module): The graph neural network module, varies based on the `model_type`.
        regressor (torch.nn.Linear): Linear layer to regress the graph-level features to target outputs.
        weight_layer (torch.nn.Sequential): MLP for L_adapt confidence vector output.
        dropout (torch.nn.Dropout): Dropout layer applied to the outputs of the GNN.

    Methods:
        forward(input): Defines the forward pass of the model using input data.
    """
    def __init__(
        self, 
        in_channels: int = MODEL_ARCHITECTURE_DEFAULTS["IN_CHANNELS"], 
        out_features: int = MODEL_ARCHITECTURE_DEFAULTS["OUT_FEATURES"],
        dropout_rate: float = MODEL_ARCHITECTURE_DEFAULTS["DROPOUT_RATE"],
        num_heads: int = MODEL_ARCHITECTURE_DEFAULTS["NUM_HEADS"],
        model_type: str = MODEL_ARCHITECTURE_DEFAULTS["MODEL_TYPE"],
        hidden_channels: int = MODEL_ARCHITECTURE_DEFAULTS["HIDDEN_CHANNELS"],
        out_channels: int = MODEL_ARCHITECTURE_DEFAULTS["OUT_CHANNELS"],
        num_layers: int = MODEL_ARCHITECTURE_DEFAULTS["NUM_LAYERS"],
        act: str='relu', 
        norm: str|NoneType=None, deg=None
    ):
        super().__init__()

        # Model definitions
        self.gnn: MLP | GCN | GAT | PNA

        if model_type == 'MLP':
            self.gnn = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type == 'GCN':
            self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type in ['GAT', 'GATv2']:
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, heads=num_heads, v2='v2' in model_type)
        elif model_type == 'PNA':
            self.gnn = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers,aggregators=['mean', 'max', 'std'], scalers=['identity'], dropout=0.0, act=act, norm=None, deg=deg, jk=None)

        self.attention_pool: AttentionalAggregation = AttentionalAggregation(gate_nn=Linear(out_channels, 1))
        # Linear layer to regress the graph-level features to target outputs
        self.regressor: Linear = Linear(in_features=out_channels, out_features=out_features)

        # MLP for L_adapt confidence vector output
        self.weight_layer: nn.Sequential = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_features)
        )

        # Dropout
        self.dropout: Dropout = torch.nn.Dropout(dropout_rate)

        # Initialize weights
        init_weights(self)

    def pooling(self, x, batch) -> Tensor:
        """Apply max pooling method."""
        return global_max_pool(x, batch)

    def forward(self, data):
        # Each snapshot contains one timestep per graph.
        # PyG batches node-wise fields as [batch_size * num_sensors].
        node_features = torch.cat(
            [
                data.x.float(),                                  # [B*N, 3]
                data.rssi.float().reshape(-1, 1),                # [B*N, 1]
                data.detected_mask.float().reshape(-1, 1),       # [B*N, 1]
                data.dist_to_wcl.float().reshape(-1, 1),         # [B*N, 1]
                data.rssi_diff_neighbor.float().reshape(-1, 1), # [B*N, 1]
            ],
            dim=1,
        )

        if node_features.size(1) != 7:
            raise RuntimeError(
                f"Expected 7 node features, got {node_features.shape}"
            )

        x = self.gnn(
            node_features,
            data.edge_index,
            edge_attr=data.edge_attr.float(),
        )

        graph_representation = self.pooling(x, data.batch)

        gnn_prediction = self.regressor(graph_representation)

        weight = torch.sigmoid(
            self.weight_layer(graph_representation)
        )

        # Batched graph-level WCL prior and target.
        wcl_pred = data.wcl_pos.float().reshape(-1, 3)

        final_prediction = (
            weight * gnn_prediction
            + (1.0 - weight) * wcl_pred
        )

        return gnn_prediction, final_prediction, weight
