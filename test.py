import torch
from torch_geometric.loader import DataLoader
from model import TSPGNN
from utils import create_full_graph, calculate_tour_distance
import os

def evaluate():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        try: import intel_extension_for_pytorch as ipex
        except ImportError: pass
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Evaluating on device: {device}")

    hidden_dim = 128
    model = TSPGNN(input_dim=2, hidden_dim=hidden_dim).to(device)
    
    model_path = "tsp_gnn_rl.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Train first!")
        return

    model.eval()

    print("Generating synthetic dataset for testing (100 graphs, 20 nodes)...")
    dataset = [create_full_graph(torch.rand((20, 2))) for _ in range(100)] 
    
    total_distance = 0
    num_samples = 0
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = data.to(device)
            
            edge_scores = model(data.x, data.edge_index, data.edge_attr)
            
            num_nodes = data.num_nodes
            visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            curr = 0
            visited[curr] = True
            
            tour_edges = []
            
            for _ in range(num_nodes - 1):
                row = data.edge_index[0]
                col = data.edge_index[1]
                
                mask = (row == curr)
                neighbors = col[mask]
                scores = edge_scores[mask]
                
                valid_mask = ~visited[neighbors]
                if valid_mask.sum() == 0: break
                
                valid_neighbors = neighbors[valid_mask]
                valid_scores = scores[valid_mask]
                
                best_idx = torch.argmax(valid_scores)
                next_node = valid_neighbors[best_idx].item()
                
                tour_edges.append((curr, next_node))
                visited[next_node] = True
                curr = next_node
                
            tour_edges.append((curr, 0))
            
            dist = calculate_tour_distance(data.x.cpu(), tour_edges)
            total_distance += dist
            num_samples += 1
            
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(dataset)} | Current Avg Distance: {total_distance/num_samples:.4f}")

    print("-" * 30)
    print(f"Final Average Tour Distance: {total_distance/num_samples:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
