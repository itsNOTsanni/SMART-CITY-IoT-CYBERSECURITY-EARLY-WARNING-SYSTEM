import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.utils.config import GNN_PARAMS, MODELS_DIR, PROPAGATION_FILE
from src.utils.logger import get_logger
from src.gnn.graph_builder import SmartCityGraphBuilder
from src.gnn.gcn_model import PropagationGCN

logger = get_logger("gnn_train")

def compute_normalized_adj(G, node_to_idx, risk_weighted=True):
    """
    Computes the risk-weighted, normalized adjacency matrix D_R^-1/2 * (A \odot R) * D_R^-1/2 with self-loops.
    This implements Layer 3 (Adaptive Graph Construction) and Layer 4 (Proposed Risk-Weighted GCN).
    """
    num_nodes = len(G.nodes())
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Weights for dynamic edge construction: A_ij = alpha * C_ij + beta * R_mean + gamma * S_ij
    alpha, beta, gamma = 0.3, 0.4, 0.3
    
    sorted_nodes = sorted(list(G.nodes()))
    node_attrs = {node: G.nodes[node] for node in sorted_nodes}
    
    # Feature extraction for similarity computation
    feats = {}
    for node in sorted_nodes:
        attrs = node_attrs[node]
        feats[node] = np.array([
            attrs['cpu_usage'] / 100.0,
            attrs['memory_usage'] / 100.0,
            attrs['traffic_rate'] / 5000.0,
            float(attrs['packet_count']) / 50000.0
        ], dtype=np.float32)
        
    for u, v in G.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # 1. Coupling / Communication Frequency
        c_ij = (node_attrs[u]['traffic_rate'] + node_attrs[v]['traffic_rate']) / 10000.0
        c_ij = min(max(c_ij, 0.0), 1.0)
        
        # 2. Risk Score of endpoints
        r_mean = (node_attrs[u]['risk_score'] + node_attrs[v]['risk_score']) / 200.0
        
        # 3. Cosine Feature Similarity
        f_u, f_v = feats[u], feats[v]
        norm_u, norm_v = np.linalg.norm(f_u), np.linalg.norm(f_v)
        s_ij = np.dot(f_u, f_v) / (norm_u * norm_v) if (norm_u > 0 and norm_v > 0) else 1.0
        
        # Adaptive dynamic edge weight
        w_ij = alpha * c_ij + beta * r_mean + gamma * s_ij
        
        # Layer 4 (Risk-weighted GCN): apply risk-masking (A \odot R)
        if risk_weighted:
            # We scale the edge propagation weight by the average attack probability of the endpoints
            p_attack_mean = (node_attrs[u]['attack_probability'] + node_attrs[v]['attack_probability']) / 2.0
            w_ij = w_ij * p_attack_mean
            
        adj[u_idx, v_idx] = w_ij
        adj[v_idx, u_idx] = w_ij
        
    # Self-loops for GCN training stability
    for node in sorted_nodes:
        idx = node_to_idx[node]
        adj[idx, idx] = 1.0
        
    # Symmetric normalization: D_R^-1/2 * (A \odot R) * D_R^-1/2
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum>0)
    d_inv_sqrt[row_sum <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
    return adj_norm

def train_gnn():
    """Trains the GNN propagation predictor model."""
    logger.info("Initializing graph builder and loading datasets...")
    graph_builder = SmartCityGraphBuilder()
    
    sorted_nodes = sorted(list(graph_builder.G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    
    # 1. Get Node Features
    X = graph_builder.get_node_features_matrix()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 2. Get Normalized Adjacency
    adj_norm = compute_normalized_adj(graph_builder.G, node_to_idx)
    adj_tensor = torch.tensor(adj_norm, dtype=torch.float32)
    
    # 3. Load Propagation samples
    if not os.path.exists(PROPAGATION_FILE):
        logger.error(f"Propagation CSV file not found at {PROPAGATION_FILE}")
        raise FileNotFoundError(f"Missing propagation CSV at {PROPAGATION_FILE}")
        
    prop_df = pd.read_csv(PROPAGATION_FILE)
    
    # Map node strings to index integers
    src_idxs = [node_to_idx[r['source']] for _, r in prop_df.iterrows()]
    tgt_idxs = [node_to_idx[r['target']] for _, r in prop_df.iterrows()]
    labels = prop_df['probability'].values
    
    src_tensor = torch.tensor(src_idxs, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_idxs, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # 4. Initialize and train across multiple seeds for statistical rigor
    import random
    seeds = [42, 43, 44, 45, 46]
    maes = []
    
    logger.info("Executing GNN training across 5 random seeds...")
    for run, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        fold_model = PropagationGCN(
            in_features=GNN_PARAMS['in_features'],
            hidden_dim=GNN_PARAMS['hidden_dim'],
            embed_dim=GNN_PARAMS['embed_dim']
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=GNN_PARAMS['lr'])
        
        fold_model.train()
        epochs = GNN_PARAMS['epochs']
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds, _ = fold_model(X_tensor, adj_tensor, src_tensor, tgt_tensor)
            loss = criterion(preds, labels_tensor)
            loss.backward()
            optimizer.step()
            
        fold_model.eval()
        with torch.no_grad():
            final_preds, _ = fold_model(X_tensor, adj_tensor, src_tensor, tgt_tensor)
            mae = torch.mean(torch.abs(final_preds - labels_tensor)).item()
        maes.append(mae)
        logger.info(f"GNN Run {run+1}/5 (Seed {seed}) MAE: {mae:.6f}")
        
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    logger.info(f"5-Seed GNN MAE: {mean_mae:.6f} ± {std_mae:.6f}")
    
    # Save the cross-validation statistics in results/
    cv_summary_path = os.path.join(os.path.dirname(MODELS_DIR), 'results', 'gnn_cv_results.txt')
    os.makedirs(os.path.dirname(cv_summary_path), exist_ok=True)
    with open(cv_summary_path, 'w') as f:
        f.write(f"GNN MAEs: {maes}\n")
        f.write(f"GNN Mean MAE: {mean_mae:.6f}\n")
        f.write(f"GNN MAE Std: {std_mae:.6f}\n")
        
    # Initialize and save the final model using seed 42
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info("Training final GNN model with baseline seed 42...")
    model = PropagationGCN(
        in_features=GNN_PARAMS['in_features'],
        hidden_dim=GNN_PARAMS['hidden_dim'],
        embed_dim=GNN_PARAMS['embed_dim']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=GNN_PARAMS['lr'])
    
    model.train()
    epochs = GNN_PARAMS['epochs']
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds, _ = model(X_tensor, adj_tensor, src_tensor, tgt_tensor)
        loss = criterion(preds, labels_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_preds, _ = model(X_tensor, adj_tensor, src_tensor, tgt_tensor)
        mae = torch.mean(torch.abs(final_preds - labels_tensor)).item()
    logger.info(f"Final GNN Training MAE: {mae:.6f}")
    
    # Save GNN Checkpoint
    save_path = os.path.join(MODELS_DIR, 'gnn_model.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'node_to_idx': node_to_idx,
        'idx_to_node': sorted_nodes,
        'adj_norm': adj_norm,
        'in_features': GNN_PARAMS['in_features'],
        'hidden_dim': GNN_PARAMS['hidden_dim'],
        'embed_dim': GNN_PARAMS['embed_dim']
    }
    
    logger.info(f"Saving GNN checkpoint to {save_path}...")
    torch.save(checkpoint, save_path)
    return model, mae


if __name__ == '__main__':
    train_gnn()
