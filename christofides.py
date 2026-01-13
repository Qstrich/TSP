import networkx as nx
import torch

def christofides_tsp(data):
    """
    Solves TSP using Christofides algorithm via NetworkX.
    data: PyTorch Geometric Data object
    Returns: tour_length (float), tour_edges (list of tuples)
    """
    # Convert PyG Data to NetworkX Graph
    G = nx.Graph()
    coords = data.x.cpu().numpy()
    num_nodes = coords.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    # Add all edges with weights
    edge_index = data.edge_index.t().cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    
    edges = []
    for i, (u, v) in enumerate(edge_index):
        weight = edge_attr[i].item()
        edges.append((u, v, weight))
        
    G.add_weighted_edges_from(edges)
    # NetworkX has a built-in approximation for Christofides
    try:
        # 'christofides' method is available in approximation module
        cycle = nx.approximation.traveling_salesman_problem(G, cycle=True, method=nx.approximation.christofides)
        
        # Calculate length
        length = 0
        tour_edges = []
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            # Find weight
            # Note: G[u][v]['weight'] might not exist if edge wasn't in original sparse graph, 
            # but we built a fully connected one so it's fine.
            dist = torch.norm(data.x[u] - data.x[v]).item()
            length += dist
            tour_edges.append((u, v))
            
        return length, tour_edges
        
    except Exception as e:
        print(f"Christofides failed: {e}")
        return float('inf'), []
