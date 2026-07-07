import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    """
    Custom Graph Convolutional Network Layer in Pure PyTorch.
    Computes: Z = D^-1/2 * A_tilde * D^-1/2 * X * W
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # adj: normalized adjacency matrix [num_nodes, num_nodes]
        # x: node features [num_nodes, in_features]
        support = self.linear(x) # [num_nodes, out_features]
        out = torch.sparse.mm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        return out

class PropagationGCN(nn.Module):
    """
    2-Layer Graph Convolutional Network model that generates node embeddings,
    and classifies link propagation risk probabilities.
    """
    def __init__(self, in_features=6, hidden_dim=16, embed_dim=8):
        super(PropagationGCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.gcn2 = GCNLayer(hidden_dim, embed_dim)
        
        # Link prediction / propagation score layer
        self.link_pred = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj, src_nodes, tgt_nodes):
        # x: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes]
        # src_nodes: indices of source nodes for propagation prediction
        # tgt_nodes: indices of target nodes for propagation prediction
        
        # 1. Generate node embeddings
        h = self.gcn1(x, adj)
        h = self.relu(h)
        h = self.gcn2(h, adj) # [num_nodes, embed_dim]
        
        # 2. Extract embeddings for pairs
        src_embeds = h[src_nodes]
        tgt_embeds = h[tgt_nodes]
        
        # 3. Combine pair representations (element-wise multiplication)
        combined = src_embeds * tgt_embeds
        
        # 4. Predict probability
        probs = self.link_pred(combined).squeeze(-1)
        
        return probs, h
