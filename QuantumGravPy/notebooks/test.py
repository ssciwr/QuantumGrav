import QuantumGrav as QGPy
import pickle


import multiprocessing


from multiprocessing import Pipe, Process

# import torch
# from torch_geometric.utils import dense_to_sparse
# from torch_geometric.data import Data


# def transform(raw) -> Data:
#     print("keys: ", raw.keys())
#     # this function will transform the raw data dictionary from Julia into a PyTorch Geometric Data object. Hence, we have to deal with julia objects here
#     adj_raw = raw["adjacency_matrix"].to_numpy()
#     adj_matrix = torch.tensor(adj_raw, dtype=torch.float32)
#     edge_index, edge_weight = dense_to_sparse(adj_matrix)
#     adj_matrix = adj_matrix.to_sparse()

#     node_features = []

#     max_path_future = torch.tensor(
#         raw["max_pathlen_future"], dtype=torch.float32
#     ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

#     max_path_past = torch.tensor(
#         raw["max_pathlen_past"], dtype=torch.float32
#     ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

#     node_features.extend([max_path_future, max_path_past])

#     # Concatenate all node features
#     x = torch.cat(node_features, dim=1)
#     classes = [raw["manifold"], raw["boundary"], raw["dimension"]]
#     y = torch.tensor(classes, dtype=torch.long)
#     data = Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_weight,
#         y=y,
#     )

#     return data


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Example usage
    onthefly_config = {
        "seed": 42,
        "n_processes": 1,
    }

    # otf_dataset = QGPy.QGDatasetOnthefly(
    #         config=onthefly_config,
    #         jl_code_path="../test/julia_testmodule.jl",
    #         jl_func_name="Generator",
    #         jl_base_module_path="../../QuantumGrav.jl",
    #         jl_dependencies=[
    #             "Distributions",
    #             "Random",
    #         ],
    #         transform=transform,
    #     )

    parent_end, child_end = Pipe()

    try:
        worker = Process(
            target=QGPy.julia_worker.worker_loop,
            args=(
                child_end,
                onthefly_config,
                "../test/julia_testmodule.jl",
                "Generator",
                None,
                "../../QuantumGrav.jl",
                [
                    "Distributions",
                    "Random",
                ],
            ),
        )

        worker.start()

        parent_end.send("GET")

        rawdata = parent_end.recv()
        data = pickle.loads(rawdata)

        worker.join()
        print(len(data), type(data[0]), flush=True)

    except Exception as e:
        print(f"Error in worker: {e}")
