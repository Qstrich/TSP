import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from model import TSPGNN
from utils import create_full_graph, calculate_tour_distance
from christofides import christofides_tsp
import os
import numpy as np
from torch.distributions import Categorical

def train_rl():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        try: import intel_extension_for_pytorch as ipex
        except ImportError: pass 
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Training on device: {device}")

    epochs = 20
    lr = 0.001
    batch_size = 32
    hidden_dim = 256
    
    model = TSPGNN(input_dim=2, hidden_dim=hidden_dim, num_layers=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Resume training if weights exist
    model_path = "tsp_gnn_rl.pth"
    if os.path.exists(model_path):
        print(f"Resuming training from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Generating synthetic data ")
    dataset = []
    for _ in range(2000):
        data = create_full_graph(torch.rand((20, 2)))
        c_len, _ = christofides_tsp(data)
        data.baseline = torch.tensor(c_len, dtype=torch.float)
        dataset.append(data)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model.train()
    print(f"Starting training")
    
    for epoch in range(epochs):
        epoch_reward = 0
        epoch_baseline = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            edge_scores_flat = model(batch.x, batch.edge_index, batch.edge_attr)
            adj_scores = to_dense_adj(batch.edge_index, batch.batch, edge_scores_flat)
            adj_scores.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
            
            B, N, _ = adj_scores.shape
            batch_coords = batch.x.view(B, N, 2)

            visited = torch.zeros((B, N), dtype=torch.bool, device=device)
            curr_nodes = torch.zeros(B, dtype=torch.long, device=device)
            visited[:, 0] = True
            
            tour_log_probs = torch.zeros(B, device=device)
            batch_rewards = torch.zeros(B, device=device)
            step_log_probs = []

            for step in range(N - 1):
                batch_indices = torch.arange(B, device=device)
                logits = adj_scores[batch_indices, curr_nodes, :]
                logits = logits.masked_fill(visited, -float('inf'))
                
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                next_nodes = dist.sample()
                
                step_log_probs.append(dist.log_prob(next_nodes))
                
                visited = visited.clone()
                visited[batch_indices, next_nodes] = True
                
                prev_coords = batch_coords[batch_indices, curr_nodes]
                new_coords = batch_coords[batch_indices, next_nodes]
                batch_rewards += torch.norm(prev_coords - new_coords, dim=1)
                
                curr_nodes = next_nodes

            last_coords = batch_coords[batch_indices, curr_nodes]
            first_coords = batch_coords[batch_indices, 0]
            batch_rewards += torch.norm(last_coords - first_coords, dim=1)
            
            tour_log_probs = torch.stack(step_log_probs).sum(dim=0)
            
            rewards = -batch_rewards
            baselines = -batch.baseline
            
            advantage = rewards - baselines
            #Normalize in start to converge faster 
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            loss = -(advantage * tour_log_probs).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_reward += batch_rewards.mean().item()
            epoch_baseline += batch.baseline.mean().item()
            
        avg_gnn = epoch_reward / len(loader)
        avg_base = epoch_baseline / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | GNN: {avg_gnn:.4f} | Base: {avg_base:.4f} | Gap: {((avg_gnn/avg_base)-1)*100:.2f}%")

    torch.save(model.state_dict(), "tsp_gnn_rl.pth")
    print("Training complete.")

if __name__ == "__main__":
    train_rl()