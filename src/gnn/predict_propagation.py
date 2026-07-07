import os
import torch
import numpy as np
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger
from src.gnn.gcn_model import PropagationGCN

logger = get_logger("gnn_predict")

class PropagationPredictor:
    def __init__(self, model_path=os.path.join(MODELS_DIR, 'gnn_model.pt')):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"GNN model checkpoint not found at {self.model_path}")
            raise FileNotFoundError(f"GNN checkpoint missing: run train_gnn.py first.")
            
        try:
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.node_to_idx = checkpoint['node_to_idx']
            self.idx_to_node = checkpoint['idx_to_node']
            self.adj_norm = checkpoint['adj_norm']
            
            self.model = PropagationGCN(
                in_features=checkpoint['in_features'],
                hidden_dim=checkpoint['hidden_dim'],
                embed_dim=checkpoint['embed_dim']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("GNN model loaded successfully.")
            self.lr_baseline = None
        except Exception as e:
            logger.error(f"Error loading GNN model checkpoint: {e}")
            raise
            
    def fit_lr_baseline(self, graph_builder):
        """Fits a logistic regression baseline using raw node features (no graph topology)."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from src.utils.config import PROPAGATION_FILE
        
        if not os.path.exists(PROPAGATION_FILE):
            logger.warning("Propagation CSV not found; skipping LR baseline fit.")
            return
            
        try:
            df = pd.read_csv(PROPAGATION_FILE)
            X_pairs = []
            y_bin = []
            
            for _, row in df.iterrows():
                src = row['source']
                tgt = row['target']
                prob = row['probability']
                
                if src in graph_builder.G and tgt in graph_builder.G:
                    f_src = [
                        graph_builder.G.nodes[src]['cpu_usage'] / 100.0,
                        graph_builder.G.nodes[src]['memory_usage'] / 100.0,
                        graph_builder.G.nodes[src]['traffic_rate'] / 5000.0,
                        float(graph_builder.G.nodes[src]['packet_count']) / 50000.0,
                        graph_builder.G.nodes[src]['attack_probability'],
                        graph_builder.G.nodes[src]['risk_score'] / 100.0
                    ]
                    f_tgt = [
                        graph_builder.G.nodes[tgt]['cpu_usage'] / 100.0,
                        graph_builder.G.nodes[tgt]['memory_usage'] / 100.0,
                        graph_builder.G.nodes[tgt]['traffic_rate'] / 5000.0,
                        float(graph_builder.G.nodes[tgt]['packet_count']) / 50000.0,
                        graph_builder.G.nodes[tgt]['attack_probability'],
                        graph_builder.G.nodes[tgt]['risk_score'] / 100.0
                    ]
                    # Element-wise product of raw attributes
                    combined = np.array(f_src) * np.array(f_tgt)
                    X_pairs.append(combined)
                    y_bin.append(1 if prob > 0.5 else 0)
                    
            if X_pairs:
                self.lr_baseline = LogisticRegression()
                self.lr_baseline.fit(np.array(X_pairs), np.array(y_bin))
                logger.info("Successfully fitted Logistic Regression baseline.")
        except Exception as e:
            logger.error(f"Failed to fit LR baseline: {e}")

    def predict_propagation(self, compromised_node, graph_builder, max_depth=3):
        """
        Simulates attack propagation using the GNN.
        Returns a list of nodes and their calculated infection probabilities.
        """
        simulate_attack = True
        if compromised_node is None:
            # Normal state evaluation: start from first node, but do not set to compromised
            compromised_node = list(graph_builder.G.nodes())[0]
            simulate_attack = False
            
        if compromised_node not in graph_builder.G:
            raise ValueError(f"Node {compromised_node} is not in the smart city graph.")
            
        if simulate_attack:
            # Temporarily simulate compromised state to update feature matrix
            graph_builder.simulate_attack_on_node(compromised_node, attack_prob=1.0, risk_score=100.0)
            
        X = graph_builder.get_node_features_matrix()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Dynamically compute the risk-weighted normalized adjacency matrix based on current state
        from src.gnn.train_gnn import compute_normalized_adj
        adj_norm = compute_normalized_adj(graph_builder.G, self.node_to_idx, risk_weighted=True)
        adj_tensor = torch.tensor(adj_norm, dtype=torch.float32)
        
        with torch.no_grad():
            # Run forward pass to extract embeddings
            h = self.model.gcn1(X_tensor, adj_tensor)
            h = self.model.relu(h)
            embeddings = self.model.gcn2(h, adj_tensor) # [num_nodes, embed_dim]
            
        # BFS propagation calculation
        start_prob = 1.0 if simulate_attack else 0.01
        visited = {compromised_node: start_prob}
        queue = [(compromised_node, start_prob, 0)] # (node_name, path_probability, current_depth)
        
        alpha, beta, gamma = 0.3, 0.4, 0.3
        
        while queue:
            curr_node, path_prob, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            curr_idx = self.node_to_idx[curr_node]
            curr_embed = embeddings[curr_idx].unsqueeze(0)
            
            neighbors = list(graph_builder.G.neighbors(curr_node))
            for nbr in neighbors:
                if nbr == compromised_node:
                    continue
                    
                nbr_idx = self.node_to_idx[nbr]
                nbr_embed = embeddings[nbr_idx].unsqueeze(0)
                
                # Combine embeddings
                combined = curr_embed * nbr_embed
                
                # Predict link probability
                with torch.no_grad():
                    link_prob = float(self.model.link_pred(combined).item())
                    
                # Calculate dynamic edge weight (Layer 3)
                c_ij = (graph_builder.G.nodes[curr_node]['traffic_rate'] + graph_builder.G.nodes[nbr]['traffic_rate']) / 10000.0
                c_ij = min(max(c_ij, 0.0), 1.0)
                r_mean = (graph_builder.G.nodes[curr_node]['risk_score'] + graph_builder.G.nodes[nbr]['risk_score']) / 200.0
                
                # Cosine feature similarity
                f_u = np.array([
                    graph_builder.G.nodes[curr_node]['cpu_usage'] / 100.0,
                    graph_builder.G.nodes[curr_node]['memory_usage'] / 100.0,
                    graph_builder.G.nodes[curr_node]['traffic_rate'] / 5000.0,
                    float(graph_builder.G.nodes[curr_node]['packet_count']) / 50000.0
                ], dtype=np.float32)
                f_v = np.array([
                    graph_builder.G.nodes[nbr]['cpu_usage'] / 100.0,
                    graph_builder.G.nodes[nbr]['memory_usage'] / 100.0,
                    graph_builder.G.nodes[nbr]['traffic_rate'] / 5000.0,
                    float(graph_builder.G.nodes[nbr]['packet_count']) / 50000.0
                ], dtype=np.float32)
                norm_u, norm_v = np.linalg.norm(f_u), np.linalg.norm(f_v)
                s_ij = np.dot(f_u, f_v) / (norm_u * norm_v) if (norm_u > 0 and norm_v > 0) else 1.0
                w_edge = alpha * c_ij + beta * r_mean + gamma * s_ij
                
                # Device susceptibility factor J(v)
                device_type = graph_builder.G.nodes[nbr].get('type', '')
                if device_type == 'ControlServer':
                    j_v = 0.7
                elif device_type == 'CCTV':
                    j_v = 1.1
                elif device_type == 'Gateway':
                    j_v = 0.9
                else:
                    j_v = 1.0
                    
                # Target node intrinsic vulnerability potential (Layer 6)
                risk_v = (graph_builder.G.nodes[nbr]['criticality'] / 10.0) * (8.5 / 10.0)
                risk_v_clamped = max(0.2, risk_v)
                
                # Weighted BFS traversal propagation probability
                link_prob = link_prob * w_edge * risk_v_clamped * j_v
                cumulative_prob = path_prob * link_prob
                
                if nbr not in visited or cumulative_prob > visited[nbr]:
                    visited[nbr] = cumulative_prob
                    queue.append((nbr, cumulative_prob, depth + 1))
                    
        # Remove seed node
        visited.pop(compromised_node, None)
        
        results = [
            {'node': k, 'probability': round(v * 100, 2)}
            for k, v in visited.items()
        ]
        return sorted(results, key=lambda x: x['probability'], reverse=True)

    def predict_propagation_shortest_path(self, compromised_node, graph_builder, max_depth=3):
        """Shortest Path Baseline: Spread decreases exponentially with hop distance."""
        import networkx as nx
        visited = {}
        for target in graph_builder.G.nodes():
            if target == compromised_node:
                continue
            try:
                dist = nx.shortest_path_length(graph_builder.G, source=compromised_node, target=target)
                if dist <= max_depth:
                    visited[target] = 0.65 ** dist
            except nx.NetworkXNoPath:
                pass
                
        results = [
            {'node': k, 'probability': round(v * 100, 2)}
            for k, v in visited.items()
        ]
        return sorted(results, key=lambda x: x['probability'], reverse=True)

    def predict_propagation_lr(self, compromised_node, graph_builder, max_depth=3):
        """Logistic Regression Baseline: Predicts spread using features only (no graph convolutions)."""
        if self.lr_baseline is None:
            self.fit_lr_baseline(graph_builder)
            if self.lr_baseline is None:
                # Fallback to shortest path if LR is not fitted
                return self.predict_propagation_shortest_path(compromised_node, graph_builder, max_depth)
                
        visited = {compromised_node: 1.0}
        queue = [(compromised_node, 1.0, 0)]
        
        while queue:
            curr_node, path_prob, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            neighbors = list(graph_builder.G.neighbors(curr_node))
            for nbr in neighbors:
                if nbr == compromised_node:
                    continue
                    
                f_src = [
                    graph_builder.G.nodes[curr_node]['cpu_usage'] / 100.0,
                    graph_builder.G.nodes[curr_node]['memory_usage'] / 100.0,
                    graph_builder.G.nodes[curr_node]['traffic_rate'] / 5000.0,
                    float(graph_builder.G.nodes[curr_node]['packet_count']) / 50000.0,
                    graph_builder.G.nodes[curr_node]['attack_probability'],
                    graph_builder.G.nodes[curr_node]['risk_score'] / 100.0
                ]
                f_tgt = [
                    graph_builder.G.nodes[nbr]['cpu_usage'] / 100.0,
                    graph_builder.G.nodes[nbr]['memory_usage'] / 100.0,
                    graph_builder.G.nodes[nbr]['traffic_rate'] / 5000.0,
                    float(graph_builder.G.nodes[nbr]['packet_count']) / 50000.0,
                    graph_builder.G.nodes[nbr]['attack_probability'],
                    graph_builder.G.nodes[nbr]['risk_score'] / 100.0
                ]
                combined = np.array(f_src) * np.array(f_tgt)
                
                # Predict using LR baseline
                link_prob = float(self.lr_baseline.predict_proba([combined])[0][1])
                cumulative_prob = path_prob * link_prob
                
                if nbr not in visited or cumulative_prob > visited[nbr]:
                    visited[nbr] = cumulative_prob
                    queue.append((nbr, cumulative_prob, depth + 1))
                    
        visited.pop(compromised_node, None)
        results = [
            {'node': k, 'probability': round(v * 100, 2)}
            for k, v in visited.items()
        ]
        return sorted(results, key=lambda x: x['probability'], reverse=True)

    def run_statistical_comparison(self, compromised_node, graph_builder):
        """Runs Wilcoxon signed-rank test comparing GNN predictions to both baselines."""
        from scipy.stats import wilcoxon
        
        gnn_res = self.predict_propagation(compromised_node, graph_builder)
        sp_res = self.predict_propagation_shortest_path(compromised_node, graph_builder)
        lr_res = self.predict_propagation_lr(compromised_node, graph_builder)
        
        # Align probabilities by node
        node_list = sorted(list(graph_builder.G.nodes()))
        node_list.remove(compromised_node)
        
        gnn_probs = {r['node']: r['probability'] for r in gnn_res}
        sp_probs = {r['node']: r['probability'] for r in sp_res}
        lr_probs = {r['node']: r['probability'] for r in lr_res}
        
        y_gnn = [gnn_probs.get(n, 0.0) for n in node_list]
        y_sp = [sp_probs.get(n, 0.0) for n in node_list]
        y_lr = [lr_probs.get(n, 0.0) for n in node_list]
        
        # Run Wilcoxon signed-rank tests
        try:
            stat_sp, p_val_sp = wilcoxon(y_gnn, y_sp)
        except Exception:
            stat_sp, p_val_sp = 0.0, 1.0 # fallback if values are identical
            
        try:
            stat_lr, p_val_lr = wilcoxon(y_gnn, y_lr)
        except Exception:
            stat_lr, p_val_lr = 0.0, 1.0
            
        logger.info(f"Statistical Test (GNN vs SP Baseline): p-value = {p_val_sp:.6f}")
        logger.info(f"Statistical Test (GNN vs LR Baseline): p-value = {p_val_lr:.6f}")
        
        return {
            'gnn_mean': np.mean(y_gnn),
            'sp_mean': np.mean(y_sp),
            'lr_mean': np.mean(y_lr),
            'p_value_vs_sp': p_val_sp,
            'p_value_vs_lr': p_val_lr
        }

    def get_critical_path(self, compromised_node, graph_builder):
        """
        Finds the critical attack path (highest probability propagation route)
        leading from the compromised node to the Control Server (or key gateways).
        """
        # We find the neighbors and evaluate probabilities step-by-step
        path = [compromised_node]
        current = compromised_node
        
        # Traverse up to 3 hops
        for _ in range(3):
            neighbors = list(graph_builder.G.neighbors(current))
            if not neighbors:
                break
                
            # Filter neighbors already in path
            candidates = [n for n in neighbors if n not in path]
            if not candidates:
                break
                
            # Predict propagation for candidates
            probs = []
            curr_idx = self.node_to_idx[current]
            
            # Extract features
            X = graph_builder.get_node_features_matrix()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            adj_tensor = torch.tensor(self.adj_norm, dtype=torch.float32)
            
            with torch.no_grad():
                h = self.model.gcn1(X_tensor, adj_tensor)
                h = self.model.relu(h)
                embeddings = self.model.gcn2(h, adj_tensor)
                
            curr_embed = embeddings[curr_idx].unsqueeze(0)
            
            for cand in candidates:
                cand_idx = self.node_to_idx[cand]
                cand_embed = embeddings[cand_idx].unsqueeze(0)
                combined = curr_embed * cand_embed
                with torch.no_grad():
                    prob = float(self.model.link_pred(combined).item())
                probs.append((cand, prob))
                
            # Pick candidate with highest probability
            probs = sorted(probs, key=lambda x: x[1], reverse=True)
            if probs and probs[0][1] > 0.15: # minimum spread threshold
                next_node = probs[0][0]
                path.append(next_node)
                current = next_node
            else:
                break
                
        return path

