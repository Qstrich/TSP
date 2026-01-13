import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class TSPGNN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 4):
        super(TSPGNN, self).__init__()
        
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=4, concat=False) 
            for _ in range(num_layers)
        ])
            
        # Input: [Node_Emb_1, Node_Emb_2, Distance]
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_embedding(x)
        
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
            
        row, col = edge_index
        
        # Concatenate Node A, Node B, and their Distance
        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        
        edge_probs = self.edge_classifier(edge_features).squeeze(-1)
        
        return edge_probs
