#!/usr/bin/env python3
"""
Simple script to check if a domain is malicious using the trained model.
This script fetches DNS records for a domain and makes predictions.
"""

import argparse
import dns.resolver
import json
from predict import DomainClassifier


def fetch_dns_records(domain: str) -> dict:
    """
    Fetch DNS records for a domain.
    
    Args:
        domain: Domain name to check
        
    Returns:
        DNS record dictionary
    """
    record = {
        "domain_name": domain,
        "dns": {
            "A": [],
            "AAAA": [],
            "MX": {},
            "NS": {},
            "TXT": [],
            "SOA": None
        }
    }
    
    # Query A records
    try:
        answers = dns.resolver.resolve(domain, 'A')
        record["dns"]["A"] = [str(rdata) for rdata in answers]
    except:
        pass
    
    # Query AAAA records
    try:
        answers = dns.resolver.resolve(domain, 'AAAA')
        record["dns"]["AAAA"] = [str(rdata) for rdata in answers]
    except:
        pass
    
    # Query MX records
    try:
        answers = dns.resolver.resolve(domain, 'MX')
        for rdata in answers:
            mx_domain = str(rdata.exchange)
            record["dns"]["MX"][mx_domain] = {
                "priority": rdata.preference,
                "related_ips": []
            }
    except:
        pass
    
    # Query NS records
    try:
        answers = dns.resolver.resolve(domain, 'NS')
        for rdata in answers:
            ns_domain = str(rdata)
            record["dns"]["NS"][ns_domain] = {
                "related_ips": []
            }
    except:
        pass
    
    # Query TXT records
    try:
        answers = dns.resolver.resolve(domain, 'TXT')
        record["dns"]["TXT"] = [str(rdata) for rdata in answers]
    except:
        pass
    
    return record


def check_domain(domain: str, model_path: str = "saved_models/best_model.pt"):
    """
    Check if a domain is malicious.
    
    Args:
        domain: Domain name to check
        model_path: Path to trained model
    """
    print(f"\n🔍 Checking domain: {domain}")
    print("=" * 60)
    
    # Fetch DNS records
    print("📡 Fetching DNS records...")
    try:
        dns_record = fetch_dns_records(domain)
        print("✓ DNS records fetched")
    except Exception as e:
        print(f"✗ Error fetching DNS records: {e}")
        return
    
    # Load model and predict
    print("🤖 Loading model and predicting...")
    try:
        classifier = DomainClassifier(
            model_path=model_path,
            scaler_path="processed_data/scaler.pkl",
            encoder_path="processed_data/label_encoder.pkl"
        )
        
        result = classifier.predict_single(dns_record)
        
        print("\n" + "=" * 60)
        print("📊 PREDICTION RESULTS")
        print("=" * 60)
        print(f"Domain: {result['domain']}")
        print(f"\n🎯 Prediction: {result['predicted_class'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"\n📈 Probabilities:")
        for cls, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"   {cls:10s}: {prob*100:5.2f}% {bar}")
        
        # Interpretation
        print("\n" + "=" * 60)
        if result['predicted_class'] == 'benign':
            print("✅ This domain appears to be SAFE")
        elif result['predicted_class'] == 'phishing':
            print("⚠️  WARNING: This domain may be used for PHISHING")
        else:  # malware
            print("🚨 ALERT: This domain may be associated with MALWARE")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Check if a domain is malicious',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_domain.py example.com
  python check_domain.py suspicious-site.tk --model saved_models/best_model.pt
  
Note: Requires dnspython package. Install with: pip install dnspython
        """
    )
    
    parser.add_argument('domain', help='Domain name to check (e.g., example.com)')
    parser.add_argument(
        '--model',
        default='saved_models/best_model.pt',
        help='Path to trained model (default: saved_models/best_model.pt)'
    )
    
    args = parser.parse_args()
    
    # Check if dnspython is installed
    try:
        import dns.resolver
    except ImportError:
        print("Error: dnspython package not found")
        print("Install it with: pip install dnspython")
        return 1
    
    check_domain(args.domain, args.model)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

