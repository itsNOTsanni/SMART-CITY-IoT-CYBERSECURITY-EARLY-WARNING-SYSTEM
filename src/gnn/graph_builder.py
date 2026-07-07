import os
import json
import networkx as nx
import numpy as np
from src.utils.config import TOPOLOGY_FILE, CRITICALITY_FILE
from src.utils.logger import get_logger

logger = get_logger("gnn_graph_builder")

class SmartCityGraphBuilder:
    def __init__(self, topology_path=TOPOLOGY_FILE, criticality_path=CRITICALITY_FILE):
        self.topology_path = topology_path
        self.criticality_path = criticality_path
        self.G = nx.Graph()
        self.load_graph()
        
    def load_graph(self):
        """Loads nodes and edges from JSON and builds the NetworkX graph."""
        if not os.path.exists(self.topology_path):
            logger.error(f"Topology file not found at {self.topology_path}")
            raise FileNotFoundError(f"Topology file not found at {self.topology_path}. Please run main.py or clean_data.py to bootstrap.")
            
        try:
            with open(self.topology_path, 'r') as f:
                topology = json.load(f)
                
            # Add nodes with attributes
            for node in topology['nodes']:
                self.G.add_node(
                    node['id'],
                    type=node['type'],
                    criticality=node['criticality'],
                    cpu_usage=float(np.random.uniform(5.0, 15.0)),
                    memory_usage=float(np.random.uniform(10.0, 25.0)),
                    traffic_rate=float(np.random.uniform(10.0, 150.0)),  # packets/sec
                    packet_count=int(np.random.randint(100, 1000)),
                    attack_probability=0.01,
                    risk_score=0.1
                )
                
            # Add edges
            for edge in topology['edges']:
                self.G.add_edge(edge['source'], edge['target'])
                
            logger.info(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error loading topology graph: {e}")
            raise
            
    def get_node_features_matrix(self):
        """Returns normalized node features as a numpy array for GNN training."""
        features = []
        sorted_nodes = sorted(list(self.G.nodes()))
        for node in sorted_nodes:
            attrs = self.G.nodes[node]
            # Feature vector normalized to [0, 1] range:
            feat = [
                attrs['cpu_usage'] / 100.0,
                attrs['memory_usage'] / 100.0,
                attrs['traffic_rate'] / 5000.0,
                float(attrs['packet_count']) / 50000.0,
                attrs['attack_probability'],
                attrs['risk_score'] / 100.0
            ]
            features.append(feat)
        return np.array(features, dtype=np.float32)

        
    def simulate_attack_on_node(self, compromised_node, attack_prob=0.95, risk_score=95.0):
        """Updates the status and attributes of a compromised node."""
        if compromised_node not in self.G:
            raise ValueError(f"Node {compromised_node} does not exist in the graph.")
            
        self.G.nodes[compromised_node]['cpu_usage'] = float(np.random.uniform(85.0, 99.0))
        self.G.nodes[compromised_node]['memory_usage'] = float(np.random.uniform(80.0, 95.0))
        self.G.nodes[compromised_node]['traffic_rate'] = float(np.random.uniform(2000.0, 5000.0))
        self.G.nodes[compromised_node]['packet_count'] = int(np.random.randint(10000, 50000))
        self.G.nodes[compromised_node]['attack_probability'] = float(attack_prob)
        self.G.nodes[compromised_node]['risk_score'] = float(risk_score)
        
    def reset_graph_states(self):
        """Resets all nodes to their normal operational baseline values."""
        for node in self.G.nodes():
            self.G.nodes[node]['cpu_usage'] = float(np.random.uniform(5.0, 15.0))
            self.G.nodes[node]['memory_usage'] = float(np.random.uniform(10.0, 25.0))
            self.G.nodes[node]['traffic_rate'] = float(np.random.uniform(10.0, 150.0))
            self.G.nodes[node]['packet_count'] = int(np.random.randint(100, 1000))
            self.G.nodes[node]['attack_probability'] = 0.01
            self.G.nodes[node]['risk_score'] = 0.1
