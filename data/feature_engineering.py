"""
Feature engineering for DNS records.
Extracts meaningful features from domain names and DNS data.
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional
import re
from collections import Counter


class DNSFeatureExtractor:
    """
    Extract features from DNS records for machine learning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names."""
        # Domain-based features
        self.feature_names.extend([
            'domain_length',
            'domain_entropy',
            'num_subdomains',
            'num_digits',
            'num_hyphens',
            'num_underscores',
            'num_special_chars',
            'digit_ratio',
            'consonant_vowel_ratio',
            'longest_word_length',
            'avg_word_length',
            'has_suspicious_tld',
            'tld_length'
        ])
        
        # DNS record count features
        self.feature_names.extend([
            'num_a_records',
            'num_aaaa_records',
            'num_mx_records',
            'num_ns_records',
            'num_txt_records',
            'has_cname',
            'has_soa',
            'total_dns_records'
        ])
        
        # IP-based features
        self.feature_names.extend([
            'num_unique_ips',
            'num_ipv4',
            'num_ipv6',
            'ip_diversity',
            'avg_ttl_ip'
        ])
        
        # TTL statistics
        self.feature_names.extend([
            'min_ttl',
            'max_ttl',
            'mean_ttl',
            'std_ttl',
            'ttl_range'
        ])
        
        # DNSSEC features
        self.feature_names.extend([
            'has_dnskey',
            'zone_dnskey_selfsign_ok',
            'dnssec_a',
            'dnssec_aaaa',
            'dnssec_soa',
            'dnssec_mx',
            'dnssec_ns'
        ])
        
        # NS/MX features
        self.feature_names.extend([
            'num_nameservers',
            'num_mail_servers',
            'has_suspicious_ns',
            'has_suspicious_mx'
        ])
        
        # SOA features
        self.feature_names.extend([
            'soa_refresh',
            'soa_retry',
            'soa_expire',
            'soa_min_ttl',
            'soa_serial'
        ])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Extract all features from a DNS record.
        
        Args:
            record: DNS record dictionary
            
        Returns:
            NumPy array of features
        """
        features = []
        
        # Extract domain name
        domain = record.get('domain_name', '')
        
        # Extract DNS data
        dns = record.get('dns', {})
        
        # 1. Domain-based features
        features.extend(self._extract_domain_features(domain))
        
        # 2. DNS record count features
        features.extend(self._extract_dns_record_counts(dns))
        
        # 3. IP-based features
        features.extend(self._extract_ip_features(dns))
        
        # 4. TTL statistics
        features.extend(self._extract_ttl_features(dns))
        
        # 5. DNSSEC features
        features.extend(self._extract_dnssec_features(dns))
        
        # 6. NS/MX features
        features.extend(self._extract_ns_mx_features(dns))
        
        # 7. SOA features
        features.extend(self._extract_soa_features(dns))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_domain_features(self, domain: str) -> List[float]:
        """Extract features from domain name."""
        features = []
        
        # Domain length
        features.append(len(domain))
        
        # Entropy
        features.append(self._calculate_entropy(domain))
        
        # Number of subdomains
        features.append(domain.count('.'))
        
        # Count digits, hyphens, underscores
        features.append(sum(c.isdigit() for c in domain))
        features.append(domain.count('-'))
        features.append(domain.count('_'))
        
        # Special characters
        special_chars = sum(not c.isalnum() and c != '.' and c != '-' and c != '_' for c in domain)
        features.append(special_chars)
        
        # Digit ratio
        digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)
        features.append(digit_ratio)
        
        # Consonant-vowel ratio
        vowels = sum(c.lower() in 'aeiou' for c in domain)
        consonants = sum(c.isalpha() and c.lower() not in 'aeiou' for c in domain)
        cv_ratio = consonants / max(vowels, 1)
        features.append(cv_ratio)
        
        # Split by dots and analyze words
        parts = domain.split('.')
        if parts:
            word_lengths = [len(p) for p in parts if p]
            features.append(max(word_lengths) if word_lengths else 0)
            features.append(np.mean(word_lengths) if word_lengths else 0)
        else:
            features.extend([0, 0])
        
        # TLD features
        tld = parts[-1] if parts else ''
        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click']
        features.append(1.0 if tld in suspicious_tlds else 0.0)
        features.append(len(tld))
        
        return features
    
    def _calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        
        counts = Counter(s)
        length = len(s)
        entropy = 0.0
        
        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _extract_dns_record_counts(self, dns: Dict) -> List[float]:
        """Extract DNS record count features."""
        features = []
        
        # A records
        a_records = dns.get('A')
        features.append(len(a_records) if a_records else 0)
        
        # AAAA records
        aaaa_records = dns.get('AAAA')
        features.append(len(aaaa_records) if aaaa_records else 0)
        
        # MX records
        mx_records = dns.get('MX')
        features.append(len(mx_records) if isinstance(mx_records, dict) else 0)
        
        # NS records
        ns_records = dns.get('NS')
        features.append(len(ns_records) if isinstance(ns_records, dict) else 0)
        
        # TXT records
        txt_records = dns.get('TXT')
        if isinstance(txt_records, list):
            features.append(len(txt_records))
        elif isinstance(txt_records, dict):
            features.append(len(txt_records))
        else:
            features.append(0)
        
        # CNAME
        features.append(1.0 if dns.get('CNAME') else 0.0)
        
        # SOA
        features.append(1.0 if dns.get('SOA') else 0.0)
        
        # Total records
        total = sum(features[:5])
        features.append(total)
        
        return features
    
    def _extract_ip_features(self, dns: Dict) -> List[float]:
        """Extract IP-based features."""
        features = []
        
        all_ips = []
        all_ttls = []
        
        # Collect IPs from A records
        a_records = dns.get('A', [])
        if a_records:
            all_ips.extend(a_records)
        
        # Collect IPs from AAAA records
        aaaa_records = dns.get('AAAA', [])
        if aaaa_records:
            all_ips.extend(aaaa_records)
        
        # Collect IPs and TTLs from MX records
        mx_records = dns.get('MX', {})
        if isinstance(mx_records, dict):
            for mx_data in mx_records.values():
                if isinstance(mx_data, dict) and 'related_ips' in mx_data:
                    for ip_entry in mx_data['related_ips']:
                        if isinstance(ip_entry, dict):
                            if 'value' in ip_entry:
                                all_ips.append(ip_entry['value'])
                            if 'ttl' in ip_entry:
                                all_ttls.append(ip_entry['ttl'])
        
        # Collect IPs and TTLs from NS records
        ns_records = dns.get('NS', {})
        if isinstance(ns_records, dict):
            for ns_data in ns_records.values():
                if isinstance(ns_data, dict) and 'related_ips' in ns_data:
                    for ip_entry in ns_data['related_ips']:
                        if isinstance(ip_entry, dict):
                            if 'value' in ip_entry:
                                all_ips.append(ip_entry['value'])
                            if 'ttl' in ip_entry:
                                all_ttls.append(ip_entry['ttl'])
        
        # Number of unique IPs
        unique_ips = len(set(all_ips)) if all_ips else 0
        features.append(unique_ips)
        
        # IPv4 vs IPv6 count
        ipv4_count = sum(1 for ip in all_ips if '.' in str(ip))
        ipv6_count = sum(1 for ip in all_ips if ':' in str(ip))
        features.append(ipv4_count)
        features.append(ipv6_count)
        
        # IP diversity (unique IPs / total IPs)
        ip_diversity = unique_ips / max(len(all_ips), 1) if all_ips else 0
        features.append(ip_diversity)
        
        # Average TTL for IPs
        avg_ttl = np.mean(all_ttls) if all_ttls else 0
        features.append(avg_ttl)
        
        return features
    
    def _extract_ttl_features(self, dns: Dict) -> List[float]:
        """Extract TTL statistics."""
        features = []
        
        all_ttls = []
        
        # Collect TTLs from various record types
        for record_type in ['MX', 'NS']:
            records = dns.get(record_type, {})
            if isinstance(records, dict):
                for record_data in records.values():
                    if isinstance(record_data, dict) and 'related_ips' in record_data:
                        for ip_entry in record_data['related_ips']:
                            if isinstance(ip_entry, dict) and 'ttl' in ip_entry:
                                all_ttls.append(ip_entry['ttl'])
        
        if all_ttls:
            features.append(min(all_ttls))
            features.append(max(all_ttls))
            features.append(np.mean(all_ttls))
            features.append(np.std(all_ttls))
            features.append(max(all_ttls) - min(all_ttls))
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def _extract_dnssec_features(self, dns: Dict) -> List[float]:
        """Extract DNSSEC features."""
        features = []
        
        remarks = dns.get('remarks', {})
        dnssec = dns.get('dnssec', {})
        
        # Has DNSKEY
        features.append(1.0 if remarks.get('has_dnskey') else 0.0)
        
        # Zone DNSKEY self-sign OK
        features.append(1.0 if remarks.get('zone_dnskey_selfsign_ok') else 0.0)
        
        # DNSSEC status for different record types
        for record_type in ['A', 'AAAA', 'SOA', 'MX', 'NS']:
            value = dnssec.get(record_type, 0)
            features.append(float(value))
        
        return features
    
    def _extract_ns_mx_features(self, dns: Dict) -> List[float]:
        """Extract nameserver and mail server features."""
        features = []
        
        # Number of nameservers
        ns_records = dns.get('NS', {})
        num_ns = len(ns_records) if isinstance(ns_records, dict) else 0
        features.append(num_ns)
        
        # Number of mail servers
        mx_records = dns.get('MX', {})
        num_mx = len(mx_records) if isinstance(mx_records, dict) else 0
        features.append(num_mx)
        
        # Check for suspicious nameservers
        suspicious_ns_keywords = ['cloudflare', 'freenom', 'afraid', 'dynamic']
        has_suspicious_ns = 0.0
        if isinstance(ns_records, dict):
            for ns_name in ns_records.keys():
                if any(keyword in ns_name.lower() for keyword in suspicious_ns_keywords):
                    has_suspicious_ns = 1.0
                    break
        features.append(has_suspicious_ns)
        
        # Check for suspicious mail servers
        has_suspicious_mx = 0.0
        if isinstance(mx_records, dict):
            for mx_name in mx_records.keys():
                if any(keyword in mx_name.lower() for keyword in suspicious_ns_keywords):
                    has_suspicious_mx = 1.0
                    break
        features.append(has_suspicious_mx)
        
        return features
    
    def _extract_soa_features(self, dns: Dict) -> List[float]:
        """Extract SOA (Start of Authority) features."""
        features = []
        
        soa = dns.get('SOA', {})
        
        if isinstance(soa, dict):
            features.append(float(soa.get('refresh', 0)))
            features.append(float(soa.get('retry', 0)))
            features.append(float(soa.get('expire', 0)))
            features.append(float(soa.get('min_ttl', 0)))
            # Normalize serial (large number)
            features.append(float(soa.get('serial', 0)) / 1e9)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def extract_features_batch(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from multiple records.
        
        Args:
            records: List of DNS record dictionaries
            
        Returns:
            NumPy array of shape (n_samples, n_features)
        """
        features_list = []
        
        for record in records:
            features = self.extract_features(record)
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)


def test_feature_extractor():
    """Test the feature extractor."""
    print("Testing DNS Feature Extractor")
    print("=" * 50)
    
    # Create a sample DNS record
    sample_record = {
        "domain_name": "example-test123.com",
        "dns": {
            "A": ["192.168.1.1", "192.168.1.2"],
            "AAAA": ["2001:db8::1"],
            "MX": {
                "mail.example.com": {
                    "priority": 10,
                    "related_ips": [
                        {"ttl": 300, "value": "192.168.1.10"}
                    ]
                }
            },
            "NS": {
                "ns1.example.com": {
                    "related_ips": [
                        {"ttl": 3600, "value": "192.168.1.20"}
                    ]
                }
            },
            "SOA": {
                "refresh": 10000,
                "retry": 2400,
                "expire": 604800,
                "min_ttl": 1800,
                "serial": 2023010101
            },
            "dnssec": {
                "A": 3,
                "AAAA": 3
            },
            "remarks": {
                "has_dnskey": True
            }
        }
    }
    
    extractor = DNSFeatureExtractor()
    
    print(f"\nNumber of features: {len(extractor.get_feature_names())}")
    print(f"\nFeature names:")
    for i, name in enumerate(extractor.get_feature_names(), 1):
        print(f"{i:2d}. {name}")
    
    print(f"\nExtracting features from sample record...")
    features = extractor.extract_features(sample_record)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    print(f"\nFeature statistics:")
    print(f"  Min: {features.min():.2f}")
    print(f"  Max: {features.max():.2f}")
    print(f"  Mean: {features.mean():.2f}")
    print(f"  Std: {features.std():.2f}")


if __name__ == "__main__":
    test_feature_extractor()

