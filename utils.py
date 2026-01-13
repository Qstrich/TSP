import torch
from torch_geometric.data import Data
import numpy as np

def create_full_graph(coords):
    num_nodes = coords.shape[0]
    adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    edge_index = np.argwhere(adj == 1).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_i = coords[edge_index[0]]
    node_j = coords[edge_index[1]]
    edge_attr = torch.norm(node_i - node_j, dim=1, keepdim=True)
    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)

def tour_to_edge_labels(tour, edge_index):
    num_edges = edge_index.shape[1]
    labels = torch.zeros(num_edges, dtype=torch.float)
    tour_edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        tour_edges.add((u, v))
        tour_edges.add((v, u))
    for i in range(num_edges):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if (u, v) in tour_edges:
            labels[i] = 1.0
    return labels

def calculate_tour_distance(coords, tour_edges):
    distance = 0
    for u, v in tour_edges:
        p1 = coords[u]
        p2 = coords[v]
        dist = torch.norm(p1 - p2).item()
        distance += dist
    return distance