import networkx as nx
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("gnn_adversarial_defense")

class GraphAttacker:
    """
    Simulates a topology poisoning attacker in a Smart City IoT network.
    The attacker attempts to inject malicious/decoy communication edges
    to bypass detection or mislead GNN early warning models.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        
    def inject_decoy_edges(self, G, poison_rate=0.2):
        """
        Injects decoy edges into the graph.
        Decoy edges are added between highly dissimilar nodes to simulate
        clandestine connections or bypass attempts (e.g. CCTV camera directly 
        linked to a Control Server without going through an Edge Gateway).
        """
        G_poisoned = G.copy()
        num_edges_to_add = int(G.number_of_edges() * poison_rate)
        nodes = list(G.nodes())
        injected_edges = []
        
        attempts = 0
        max_attempts = num_edges_to_add * 10
        
        while len(injected_edges) < num_edges_to_add and attempts < max_attempts:
            attempts += 1
            u, v = self.rng.choice(nodes, size=2, replace=False)
            
            # Avoid adding self-loops or duplicate edges
            if G_poisoned.has_edge(u, v):
                continue
                
            u_type = G.nodes[u].get('type', '')
            v_type = G.nodes[v].get('type', '')
            
            # Attacker prioritizes stealthy shortcuts: edge device directly to core servers
            is_stealthy_target = (
                (u_type in ['CCTV', 'TrafficLight', 'SmartMeter', 'SmartParking'] and v_type == 'ControlServer') or
                (v_type in ['CCTV', 'TrafficLight', 'SmartMeter', 'SmartParking'] and u_type == 'ControlServer')
            )
            
            # If not stealthy target, inject with 30% probability to keep some randomness
            if not is_stealthy_target and self.rng.random() > 0.3:
                continue
                
            G_poisoned.add_edge(u, v, is_adversarial=True)
            injected_edges.append((u, v))
            
        logger.info(f"Adversarial topology attack completed: Injected {len(injected_edges)} decoy edges.")
        return G_poisoned, injected_edges


class GraphPurifier:
    """
    Implements Graph Adversarial Purification (GAP) using Jaccard & Cosine similarity
    metrics to filter out anomalous and suspicious communication paths.
    """
    def __init__(self, threshold=0.45, w_t=0.5, w_f=0.5):
        self.threshold = threshold
        self.w_t = w_t  # Weight for type semantic similarity
        self.w_f = w_f  # Weight for feature telemetry similarity
        
        # Define semantic similarity between smart city device categories
        # High value = authorized/standard communication path
        # Low value = abnormal/suspicious/direct backdoor path
        self.type_sim_matrix = {
            ('CCTV', 'EdgeGateway'): 0.85,
            ('TrafficLight', 'EdgeGateway'): 0.85,
            ('SmartMeter', 'EdgeGateway'): 0.85,
            ('SmartParking', 'EdgeGateway'): 0.85,
            ('EdgeGateway', 'ControlServer'): 0.95,
            ('ControlServer', 'ControlServer'): 0.80,
            ('EdgeGateway', 'EdgeGateway'): 0.50,
            ('CCTV', 'CCTV'): 0.60,
            ('TrafficLight', 'TrafficLight'): 0.60,
            # Malicious cross-domain or shortcut pathways
            ('CCTV', 'ControlServer'): 0.05,
            ('TrafficLight', 'ControlServer'): 0.05,
            ('SmartMeter', 'ControlServer'): 0.05,
            ('SmartParking', 'ControlServer'): 0.05,
            ('CCTV', 'TrafficLight'): 0.15,
            ('CCTV', 'SmartMeter'): 0.15,
        }

    def _get_type_similarity(self, type1, type2):
        if type1 == type2:
            return self.type_sim_matrix.get((type1, type2), 0.5)
        # Check bidirectional pairs
        pair = (type1, type2)
        rev_pair = (type2, type1)
        if pair in self.type_sim_matrix:
            return self.type_sim_matrix[pair]
        if rev_pair in self.type_sim_matrix:
            return self.type_sim_matrix[rev_pair]
        # Default for unmapped normal communication paths
        return 0.3

    def _calculate_cosine_similarity(self, G, u, v):
        """Calculates cosine similarity of telemetry features between node u and v."""
        # Feature list: [cpu_usage, memory_usage, traffic_rate, packet_count, attack_prob, risk_score]
        def get_features(node):
            attrs = G.nodes[node]
            return np.array([
                attrs.get('cpu_usage', 10.0),
                attrs.get('memory_usage', 20.0),
                attrs.get('traffic_rate', 50.0),
                float(attrs.get('packet_count', 500)),
                attrs.get('attack_probability', 0.01),
                attrs.get('risk_score', 0.1)
            ])
            
        feat_u = get_features(u)
        feat_v = get_features(v)
        
        # Min-max normalize elements to prevent scale dominance (packet_count / traffic_rate vs risk)
        scales = np.array([100.0, 100.0, 5000.0, 50000.0, 1.0, 100.0])
        feat_u_norm = feat_u / scales
        feat_v_norm = feat_v / scales
        
        dot_product = np.dot(feat_u_norm, feat_v_norm)
        norm_u = np.linalg.norm(feat_u_norm)
        norm_v = np.linalg.norm(feat_v_norm)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
            
        return float(dot_product / (norm_u * norm_v))

    def compute_edge_similarity(self, G, u, v):
        """Computes hybrid similarity score for a communication link."""
        u_type = G.nodes[u].get('type', '')
        v_type = G.nodes[v].get('type', '')
        
        sim_type = self._get_type_similarity(u_type, v_type)
        sim_feat = self._calculate_cosine_similarity(G, u, v)
        
        # Hybrid formulation
        hybrid_score = self.w_t * sim_type + self.w_f * sim_feat
        return hybrid_score

    def purify_topology(self, G_poisoned, custom_threshold=None):
        """
        Prunes communication edges with similarity scores below the threshold.
        Returns the purified graph, a list of pruned edges, and a dictionary of similarity scores.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        G_purified = G_poisoned.copy()
        pruned_edges = []
        edge_similarities = {}
        
        for u, v in list(G_poisoned.edges()):
            score = self.compute_edge_similarity(G_poisoned, u, v)
            edge_similarities[(u, v)] = score
            
            if score < threshold:
                G_purified.remove_edge(u, v)
                pruned_edges.append((u, v))
                
        logger.info(f"Purification completed: Pruned {len(pruned_edges)} suspicious edges using threshold={threshold}.")
        return G_purified, pruned_edges, edge_similarities


class RobustnessEvaluator:
    """
    Evaluates the vulnerability and robustness metrics of the early warning system.
    """
    @staticmethod
    def evaluate_defense(injected_edges, pruned_edges):
        """
        Computes precision, recall, and F1-score of the purifier in detecting injected decoy links.
        """
        injected_set = set(tuple(sorted(e)) for e in injected_edges)
        pruned_set = set(tuple(sorted(e)) for e in pruned_edges)
        
        tp = len(injected_set.intersection(pruned_set)) # True Positives: correctly pruned decoy edges
        fp = len(pruned_set - injected_set)             # False Positives: normal edges mistakenly pruned
        fn = len(injected_set - pruned_set)             # False Negatives: decoy edges missed
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2)
        }
        
    @staticmethod
    def compute_accuracy_restoration(clean_metrics, poisoned_metrics, purified_metrics):
        """
        Computes the Defense Recovery Index (DRI), indicating what percentage of 
        GNN prediction capability was restored compared to the unperturbed baseline.
        """
        # Let's use the average prediction error compared to baseline
        # In a real environment, we check differences in propagation path overlap or critical warning accuracy.
        # Here we compute MSE discrepancy between GNN outputs.
        pass
