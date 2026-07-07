from src.utils.logger import get_logger

logger = get_logger("early_warning_alerts")

class AlertGenerator:
    def __init__(self):
        pass
        
    def generate_alerts(self, risk_data, attack_type):
        """
        Generates security alerts, warning actions, and response checklists
        based on the risk level and attack class.
        """
        node_id = risk_data['node_id']
        risk_level = risk_data['risk_level']
        risk_score = risk_data['risk_score']
        
        recommendations = []
        status_msg = ""
        color = "#ECEFF1" # standard grey
        
        if attack_type == 'Normal':
            status_msg = "NORMAL OPERATIONS"
            color = "#66BB6A" # green
            recommendations = ["✓ Continuous network monitoring active"]
            return {
                'status': status_msg,
                'color': color,
                'alerts': ["System operational. No threats detected."],
                'recommendations': recommendations
            }
            
        # Map colors for threat levels
        if risk_level == 'LOW':
            color = "#81C784" # light green
            status_msg = "ADVISORY - LOW THREAT DETECTED"
        elif risk_level == 'MEDIUM':
            color = "#FFD54F" # yellow
            status_msg = "WARNING - MEDIUM LEVEL THREAT"
        elif risk_level == 'HIGH':
            color = "#FFA726" # orange
            status_msg = "ALERT - HIGH LEVEL THREAT DETECTED"
        else: # CRITICAL
            color = "#EF5350" # red
            status_msg = "IMMEDIATE ACTION REQUIRED"
            
        alerts = [
            f"Alert: {attack_type} detected on device '{node_id}'!",
            f"Early Warning System flags risk score of {risk_score}/100 ({risk_level})."
        ]
        
        # Recommendations based on attack type
        if attack_type == 'DDoS':
            recommendations = [
                f"✓ Isolate compromised {node_id} from local network segment.",
                "✓ Block source IP addresses at primary firewall gateway.",
                "✓ Enable traffic rate limiting and blackholing at perimeter router.",
                "✓ Divert critical API requests to redundant control nodes."
            ]
        elif attack_type == 'DoS':
            recommendations = [
                f"✓ Rate limit incoming traffic at gateway for {node_id}.",
                "✓ Enable SYN cookie protection on target server/node.",
                "✓ Block attacker source IP addresses.",
                "✓ Check server logs for specific TCP handshake anomalies."
            ]
        elif attack_type == 'Botnet':
            recommendations = [
                f"✓ Isolate gateway connected to {node_id}.",
                "✓ Block active Command & Control (C&C) server IPs.",
                "✓ Terminate suspicious persistent sockets and connections.",
                "✓ Disconnect affected sensors and flash firmware."
            ]
        elif attack_type == 'Brute Force':
            recommendations = [
                f"✓ Lock administrative accounts on {node_id} temporarily.",
                "✓ Block source IP address after multiple failed attempts.",
                "✓ Enforce multi-factor authentication (MFA) and rotate SSH keys.",
                "✓ Update network access list rules (ACL) for management interfaces."
            ]
        elif attack_type == 'Spoofing':
            recommendations = [
                f"✓ Block MAC address of spoofing device on network switch.",
                "✓ Refresh ARP table caches on edge routers and control nodes.",
                "✓ Implement 802.1X port-based access control authentication.",
                "✓ Check configurations for rogue DHCP/ARP servers."
            ]
        elif attack_type == 'Reconnaissance':
            recommendations = [
                "✓ Block source scanning IP address.",
                "✓ Disable unused open ports and services on edge gateways.",
                "✓ Enable intrusion prevention system (IPS) port scan defenses.",
                "✓ Update firewall logging configurations."
            ]
        else:
            recommendations = [
                f"✓ Inspect traffic profiles on {node_id}.",
                "✓ Verify device firmware signature integrity."
            ]
            
        # Additional emergency checklist for critical threats
        if risk_level in ['HIGH', 'CRITICAL']:
            recommendations.extend([
                "✓ Alert Security Operations Center (SOC) on-call personnel.",
                "✓ Trigger incident response containment playbooks.",
                "✓ Log network packet traces for forensic audits."
            ])
            
        logger.info(f"Generated {len(recommendations)} alerts and response items.")
        
        return {
            'status': status_msg,
            'color': color,
            'alerts': alerts,
            'recommendations': recommendations
        }
