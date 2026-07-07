import os
import json
import pandas as pd
import numpy as np

def generate_ciciot2023(num_samples=5000):
    """
    Generates synthetic CICIoT2023 dataset containing:
    DDoS, DoS, Spoofing, Botnet, Brute Force, Reconnaissance, Normal.
    """
    np.random.seed(42)
    classes = ['Normal', 'DDoS', 'DoS', 'Spoofing', 'Botnet', 'Brute Force', 'Reconnaissance']
    probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10]
    labels = np.random.choice(classes, size=num_samples, p=probs)
    
    data = []
    for label in labels:
        # Features: Flow Duration, Packet Count, Protocol, Source Port, Destination Port, Flow Bytes, Packet Rate
        if label == 'Normal':
            flow_duration = np.random.uniform(0.5, 30.0)
            packet_count = np.random.randint(5, 100)
            protocol = np.random.choice([6, 17, 1]) # TCP, UDP, ICMP
            src_port = np.random.choice([80, 443, 53, 123] + list(range(1024, 65535)))
            dst_port = np.random.choice([80, 443, 53, 123] + list(range(1024, 65535)))
            flow_bytes = packet_count * np.random.uniform(40, 1000)
            packet_rate = packet_count / flow_duration
        elif label == 'DDoS':
            flow_duration = np.random.uniform(0.01, 1.5)
            packet_count = np.random.randint(1000, 5000)
            protocol = np.random.choice([17, 1]) # UDP or ICMP flood
            src_port = np.random.randint(1024, 65535)
            dst_port = np.random.choice([80, 443, 22])
            flow_bytes = packet_count * np.random.uniform(64, 128) # small identical packets
            packet_rate = packet_count / flow_duration
        elif label == 'DoS':
            flow_duration = np.random.uniform(5.0, 60.0)
            packet_count = np.random.randint(500, 2000)
            protocol = 6 # TCP (Syn flood)
            src_port = np.random.randint(1024, 65535)
            dst_port = np.random.choice([80, 443, 8080])
            flow_bytes = packet_count * np.random.uniform(40, 64)
            packet_rate = packet_count / flow_duration
        elif label == 'Spoofing':
            flow_duration = np.random.uniform(0.1, 5.0)
            packet_count = np.random.randint(10, 150)
            protocol = np.random.choice([1, 6])
            src_port = np.random.randint(1, 1024)
            dst_port = np.random.randint(1, 65535)
            flow_bytes = packet_count * np.random.uniform(40, 200)
            packet_rate = packet_count / flow_duration
        elif label == 'Botnet':
            # periodic, low and slow, specific patterns
            flow_duration = np.random.uniform(2.0, 45.0)
            packet_count = np.random.randint(15, 200)
            protocol = 6
            src_port = np.random.randint(10000, 65535)
            dst_port = np.random.choice([6667, 8080, 9999]) # typical IRC/C&C ports
            flow_bytes = packet_count * np.random.uniform(60, 400)
            packet_rate = packet_count / flow_duration
        elif label == 'Brute Force':
            flow_duration = np.random.uniform(1.0, 15.0)
            packet_count = np.random.randint(100, 600)
            protocol = 6
            src_port = np.random.randint(1024, 65535)
            dst_port = np.random.choice([22, 23, 3389]) # SSH, Telnet, RDP
            flow_bytes = packet_count * np.random.uniform(80, 300)
            packet_rate = packet_count / flow_duration
        elif label == 'Reconnaissance':
            flow_duration = np.random.uniform(0.05, 3.0)
            packet_count = np.random.randint(10, 300)
            protocol = np.random.choice([6, 17])
            src_port = np.random.randint(1024, 65535)
            dst_port = np.random.randint(1, 1024) # scanning system ports
            flow_bytes = packet_count * 40 # tiny packets (SYN scan)
            packet_rate = packet_count / flow_duration

        data.append({
            'flow_duration': flow_duration,
            'packet_count': packet_count,
            'protocol': protocol,
            'src_port': src_port,
            'dst_port': dst_port,
            'flow_bytes': flow_bytes,
            'packet_rate': packet_rate,
            'label': label
        })
        
    df = pd.DataFrame(data)
    os.makedirs('datasets/CICIoT2023', exist_ok=True)
    df.to_csv('datasets/CICIoT2023/CICIoT2023_subset.csv', index=False)
    print(f"Generated {num_samples} samples for CICIoT2023.")

def generate_ton_iot(num_samples=2000):
    """
    Generates synthetic TON_IoT dataset containing:
    Weather sensors, GPS trackers, smart fridges, etc.
    Attacks: Normal, DDoS, Backdoor, Injection, Password.
    """
    np.random.seed(43)
    devices = ['WeatherSensor', 'GPSTracker', 'SmartFridge', 'SmartGarageDoor']
    attacks = ['Normal', 'DDoS', 'Backdoor', 'Injection', 'Password']
    
    data = []
    for _ in range(num_samples):
        device = np.random.choice(devices)
        attack = np.random.choice(attacks, p=[0.4, 0.2, 0.15, 0.15, 0.1])
        
        # Simulating sensor features: temperature, humidity, memory_usage, cpu_usage, packet_rate
        cpu_usage = np.random.uniform(1.0, 15.0) if attack == 'Normal' else np.random.uniform(40.0, 99.0)
        memory_usage = np.random.uniform(5.0, 30.0) if attack == 'Normal' else np.random.uniform(50.0, 95.0)
        
        if device == 'WeatherSensor':
            temp = np.random.uniform(15.0, 35.0) if attack != 'Injection' else np.random.uniform(-100, 500) # outlier injection
            humidity = np.random.uniform(30.0, 90.0)
            status = 'normal' if attack == 'Normal' else 'anomaly'
        elif device == 'GPSTracker':
            temp = np.random.uniform(25.0, 45.0)
            humidity = np.random.uniform(10.0, 50.0)
            status = 'normal' if attack == 'Normal' else 'anomaly'
        else: # fridge/garage
            temp = np.random.uniform(2.0, 8.0) if device == 'SmartFridge' else np.random.uniform(15.0, 25.0)
            humidity = np.random.uniform(40.0, 60.0)
            status = 'normal' if attack == 'Normal' else 'anomaly'
            
        data.append({
            'device_type': device,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'temperature': temp,
            'humidity': humidity,
            'status': status,
            'label': attack
        })
        
    df = pd.DataFrame(data)
    os.makedirs('datasets/IoT23', exist_ok=True) # creating parent directories
    os.makedirs('datasets/TON_IoT', exist_ok=True)
    df.to_csv('datasets/TON_IoT/TON_IoT_subset.csv', index=False)
    print(f"Generated {num_samples} samples for TON_IoT.")

def generate_iot23(num_samples=2000):
    """
    Generates synthetic IoT-23 dataset containing:
    Mirai Botnet, Malware, C&C Traffic, Normal.
    """
    np.random.seed(44)
    labels = np.random.choice(['Normal', 'Mirai', 'Malware', 'C&C'], size=num_samples, p=[0.3, 0.3, 0.2, 0.2])
    
    data = []
    for label in labels:
        # Features: duration, orig_bytes, resp_bytes, conn_state, history
        if label == 'Normal':
            duration = np.random.uniform(0.1, 10.0)
            orig_bytes = np.random.randint(100, 5000)
            resp_bytes = np.random.randint(100, 20000)
            conn_state = 'SF' # Established and terminated cleanly
            history = 'ShADadfF'
        elif label == 'Mirai':
            duration = np.random.uniform(0.01, 0.5)
            orig_bytes = np.random.randint(500, 1500)
            resp_bytes = 0 # Syn flooding
            conn_state = 'S0' # Connection attempt seen, no reply
            history = 'S'
        elif label == 'Malware':
            duration = np.random.uniform(5.0, 120.0)
            orig_bytes = np.random.randint(5000, 100000)
            resp_bytes = np.random.randint(1000, 50000)
            conn_state = 'OTH'
            history = 'D'
        else: # C&C
            duration = np.random.uniform(60.0, 3600.0) # long lived connections
            orig_bytes = np.random.randint(50, 1000)   # heartbeat packets (small)
            resp_bytes = np.random.randint(50, 1000)
            conn_state = 'S1'
            history = 'ShAdDaf'
            
        data.append({
            'duration': duration,
            'orig_bytes': orig_bytes,
            'resp_bytes': resp_bytes,
            'conn_state': conn_state,
            'history': history,
            'label': label
        })
        
    df = pd.DataFrame(data)
    df.to_csv('datasets/IoT23/IoT23_subset.csv', index=False)
    print(f"Generated {num_samples} samples for IoT-23.")

def generate_smart_city_topology_and_assets():
    """
    Generates a 100-node smart city network topology.
    Saves graph nodes and edges to JSON, and asset criticality to CSV.
    """
    np.random.seed(45)
    
    # Define node types
    node_types = {
        'CCTV': 30,
        'TrafficLight': 30,
        'ParkingSensor': 15,
        'SmartMeter': 10,
        'WaterSensor': 5,
        'StreetLight': 5,
        'Gateway': 3,
        'Server': 2
    }
    
    # Asset criticality mapping (scale of 1 to 10)
    criticality_map = {
        'Server': 10,
        'Gateway': 9,
        'TrafficLight': 8,
        'WaterSensor': 7,
        'CCTV': 6,
        'SmartMeter': 4,
        'ParkingSensor': 3,
        'StreetLight': 2
    }
    
    # Create nodes list
    nodes = []
    node_by_type = {}
    for ntype, count in node_types.items():
        node_by_type[ntype] = []
        for i in range(1, count + 1):
            name = f"{ntype}_{i}"
            nodes.append({
                'id': name,
                'type': ntype,
                'criticality': criticality_map[ntype]
            })
            node_by_type[ntype].append(name)
            
    # Save Asset Criticality Dataset
    asset_df = pd.DataFrame(nodes)
    asset_df.to_csv('datasets/asset_criticality.csv', index=False)
    
    # Generate edges representing topology
    # CCTVs, Lights, Sensors connect to Gateways
    # Gateways connect to Servers
    edges = []
    gateways = node_by_type['Gateway']
    servers = node_by_type['Server']
    
    # Server-to-Server connection
    edges.append({'source': servers[0], 'target': servers[1]})
    
    # Gateway-to-Server connections
    for i, gw in enumerate(gateways):
        # connect each gateway to at least one server
        srv = servers[i % len(servers)]
        edges.append({'source': gw, 'target': srv})
        
    # Connect IoT edge devices to Gateways
    edge_types = ['CCTV', 'TrafficLight', 'ParkingSensor', 'SmartMeter', 'WaterSensor', 'StreetLight']
    for etype in edge_types:
        for idx, device in enumerate(node_by_type[etype]):
            # connect each device to one gateway based on index
            gw = gateways[idx % len(gateways)]
            edges.append({'source': device, 'target': gw})
            
            # also introduce some horizontal connections (e.g. adjacent traffic lights or CCTVs)
            if idx > 0 and np.random.rand() < 0.2:
                edges.append({'source': node_by_type[etype][idx-1], 'target': device})
                
    # Save Smart City Topology Dataset
    topology = {
        'nodes': nodes,
        'edges': edges
    }
    with open('datasets/smart_city_topology.json', 'w') as f:
        json.dump(topology, f, indent=4)
        
    print(f"Generated smart city topology: {len(nodes)} nodes, {len(edges)} edges.")
    
    # Generate Attack Propagation Dataset
    # Maps typical propagation probabilities between device types
    propagation_rules = [
        {'source_type': 'CCTV', 'target_type': 'Gateway', 'probability': 0.85},
        {'source_type': 'TrafficLight', 'target_type': 'Gateway', 'probability': 0.80},
        {'source_type': 'ParkingSensor', 'target_type': 'Gateway', 'probability': 0.65},
        {'source_type': 'SmartMeter', 'target_type': 'Gateway', 'probability': 0.70},
        {'source_type': 'WaterSensor', 'target_type': 'Gateway', 'probability': 0.75},
        {'source_type': 'StreetLight', 'target_type': 'Gateway', 'probability': 0.55},
        {'source_type': 'Gateway', 'target_type': 'Server', 'probability': 0.90},
        {'source_type': 'Server', 'target_type': 'Gateway', 'probability': 0.85},
        {'source_type': 'Server', 'target_type': 'Server', 'probability': 0.95},
        # Horizontal propagation
        {'source_type': 'CCTV', 'target_type': 'CCTV', 'probability': 0.75},
        {'source_type': 'TrafficLight', 'target_type': 'TrafficLight', 'probability': 0.70},
        {'source_type': 'TrafficLight', 'target_type': 'CCTV', 'probability': 0.60},
        {'source_type': 'CCTV', 'target_type': 'TrafficLight', 'probability': 0.60},
    ]
    
    # Create custom node-to-node propagation probabilities based on actual links
    prop_data = []
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        src_type = src.split('_')[0]
        tgt_type = tgt.split('_')[0]
        
        # Match probability
        prob = 0.50 # default
        for rule in propagation_rules:
            if (rule['source_type'] == src_type and rule['target_type'] == tgt_type) or \
               (rule['source_type'] == tgt_type and rule['target_type'] == src_type):
                prob = rule['probability']
                break
        
        # Add slight random noise to look realistic
        prob = float(np.clip(prob + np.random.normal(0, 0.03), 0.1, 0.99))
        
        prop_data.append({'source': src, 'target': tgt, 'probability': prob})
        prop_data.append({'source': tgt, 'target': src, 'probability': prob}) # bi-directional spread
        
    prop_df = pd.DataFrame(prop_data)
    prop_df.to_csv('datasets/attack_propagation.csv', index=False)
    print(f"Generated attack propagation dataset: {len(prop_data)} paths.")
    
    # Generate Impact Dataset
    # Maps Attack Type, Asset Type -> Service Disruption %
    impact_data = [
        {'attack_type': 'DDoS', 'asset_type': 'Server', 'impact': 0.95},
        {'attack_type': 'DDoS', 'asset_type': 'Gateway', 'impact': 0.85},
        {'attack_type': 'DDoS', 'asset_type': 'TrafficLight', 'impact': 0.60},
        {'attack_type': 'DDoS', 'asset_type': 'CCTV', 'impact': 0.40},
        {'attack_type': 'Botnet', 'asset_type': 'Server', 'impact': 0.70},
        {'attack_type': 'Botnet', 'asset_type': 'Gateway', 'impact': 0.60},
        {'attack_type': 'Botnet', 'asset_type': 'TrafficLight', 'impact': 0.50},
        {'attack_type': 'Botnet', 'asset_type': 'CCTV', 'impact': 0.80},
        {'attack_type': 'Spoofing', 'asset_type': 'TrafficLight', 'impact': 0.90},
        {'attack_type': 'Spoofing', 'asset_type': 'CCTV', 'impact': 0.30},
        {'attack_type': 'Spoofing', 'asset_type': 'ParkingSensor', 'impact': 0.50},
        {'attack_type': 'DoS', 'asset_type': 'Server', 'impact': 0.90},
        {'attack_type': 'DoS', 'asset_type': 'Gateway', 'impact': 0.80},
        {'attack_type': 'DoS', 'asset_type': 'TrafficLight', 'impact': 0.70},
        {'attack_type': 'Brute Force', 'asset_type': 'Server', 'impact': 0.50},
        {'attack_type': 'Reconnaissance', 'asset_type': 'Server', 'impact': 0.15},
    ]
    
    # Fill in the rest dynamically with baseline defaults
    all_attacks = ['Normal', 'DDoS', 'DoS', 'Spoofing', 'Botnet', 'Brute Force', 'Reconnaissance']
    all_assets = list(criticality_map.keys())
    
    full_impact_data = []
    existing_pairs = {(item['attack_type'], item['asset_type']) for item in impact_data}
    
    for item in impact_data:
        full_impact_data.append(item)
        
    for atk in all_attacks:
        for ast in all_assets:
            if (atk, ast) not in existing_pairs:
                if atk == 'Normal':
                    impact = 0.0
                else:
                    # heuristic
                    impact = (criticality_map[ast] / 10.0) * 0.25
                full_impact_data.append({'attack_type': atk, 'asset_type': ast, 'impact': float(impact)})
                
    impact_df = pd.DataFrame(full_impact_data)
    impact_df.to_csv('datasets/impact_assessment.csv', index=False)
    print(f"Generated impact assessment dataset with {len(full_impact_data)} mappings.")

if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)
    generate_ciciot2023()
    generate_ton_iot()
    generate_iot23()
    generate_smart_city_topology_and_assets()
    print("All synthetic datasets generated successfully!")
