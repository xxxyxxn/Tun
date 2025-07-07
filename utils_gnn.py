import torch
import numpy as np

def convert_to_gnn_input(prob_info):
    N = prob_info["N"]
    E = prob_info["E"]
    nodes_info = prob_info["grid_graph"]["nodes"]

    # 1. edge_index (양방향)
    edge_list = E + [[to, frm] for frm, to in E]
    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    # 2. 노드 feature x
    type_map = {"gate": [1, 0, 0], "hold": [0, 1, 0], "ramp": [0, 0, 1]}
    x_list = [None] * N

    for coord, info in nodes_info:
        node_id = info["id"]
        type_onehot = type_map[info["type"]]
        distance = info["distance"]
        x_list[node_id] = type_onehot + [distance]

    for i in range(N):
        if x_list[i] is None:
            x_list[i] = [0.0, 0.0, 0.0, 0.0]

    x = torch.tensor(x_list, dtype=torch.float)

    return x, edge_index
