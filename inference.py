import torch
import matplotlib.pyplot as plt
import networkx as nx
from model import TSPGNN
from utils import create_full_graph

def greedy_tour(coords, edge_index, edge_probs):
    num_nodes = coords.shape[0]
    edges = []
    for i in range(edge_index.shape[1]):
        edges.append((edge_index[0, i].item(), edge_index[1, i].item(), edge_probs[i].item()))
        
    edges.sort(key=lambda x: x[2], reverse=True)
    
    selected_edges = []
    adj = {i: [] for i in range(num_nodes)}
    
    for u, v, p in edges:
        if u < v and len(adj[u]) < 2 and len(adj[v]) < 2:
            selected_edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if len(selected_edges) == num_nodes:
            break
            
    return selected_edges

def visualize_tsp(coords, selected_edges):
    plt.figure(figsize=(8, 6))
    coords_np = coords.numpy()
    
    plt.scatter(coords_np[:, 0], coords_np[:, 1], c='red', zorder=5)
    for i, (x, y) in enumerate(coords_np):
        plt.text(x, y, str(i), fontsize=12)
        
    for u, v in selected_edges:
        p1 = coords_np[u]
        p2 = coords_np[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='blue', alpha=0.6)
        
    plt.title("GNN Predicted TSP Tour (Greedy Reconstruction)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def run_inference():
    num_nodes = 20
    model = TSPGNN(input_dim=2, hidden_dim=64)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("tsp_gnn.pth"))
    else:
        model.load_state_dict(torch.load("tsp_gnn.pth", map_location=torch.device('cpu')))
        
    model.eval()
    
    coords = torch.rand((num_nodes, 2))
    data = create_full_graph(coords)
    
    with torch.no_grad():
        probs = model(data.x, data.edge_index)
    
    tour_edges = greedy_tour(coords, data.edge_index, probs)
    print(f"Selected {len(tour_edges)} edges for the tour.")
    
    visualize_tsp(coords, tour_edges)

if __name__ == "__main__":
    import os
    if os.path.exists("tsp_gnn.pth"):
        run_inference()
    else:
        print("Model file 'tsp_gnn.pth' not found. Please run 'train.py' first.")
