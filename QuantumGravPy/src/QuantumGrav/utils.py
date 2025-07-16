import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from typing import Any


def transform(raw: dict[Any, Any]) -> dict[Any, Any]:
    # get adjacency matrix and convert it to edge_index and edge_weight
    adj_raw = raw["adjacency_matrix"]
    adj_matrix = torch.tensor(adj_raw, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adj_matrix)
    adj_matrix = adj_matrix.to_sparse()

    # Load node features. We are only using degree information and path lengths for now.
    node_features = []

    # Degree information
    in_degrees = torch.tensor(raw["in_degrees"], dtype=torch.float32).unsqueeze(1)
    out_degrees = torch.tensor(raw["out_degrees"], dtype=torch.float32).unsqueeze(1)
    node_features.extend([in_degrees, out_degrees])

    # Path lengths
    max_path_future = torch.tensor(
        raw["max_path_lengths_future"], dtype=torch.float32
    ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

    max_path_past = torch.tensor(
        raw["max_path_lengths_past"], dtype=torch.float32
    ).unsqueeze(1)  # make this a (num_nodes, 1) tensor
    node_features.extend([max_path_future, max_path_past])

    # Concatenate all node features
    x = torch.cat(node_features, dim=1)

    # create the Data object
    datapoint = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1)
        if edge_weight.numel() > 0
        else None,  # Not sure if this is a good idea need to add edge attributes if possible
        # the classification target. This concatenates everything into a single 1D tensor.
        y=torch.tensor(
            [raw["manifold"], raw["boundary"], raw["dimension"]], dtype=torch.long
        ),
    )

    datapoint.validate()

    return datapoint
