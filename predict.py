"""
Prediction script for malicious domain detection.
Loads a trained model and makes predictions on new domains.
PyTorch implementation.
"""

import os
import numpy as np
import argparse
import json
from typing import List, Dict, Any
import torch
import torch.nn.functional as F

from data.feature_engineering import DNSFeatureExtractor
from data.preprocessor import DNSDataPreprocessor, prepare_data_for_cnn
from models.cnn_lstm_model import build_cnn_lstm_model


class DomainClassifier:
    """
    Domain classifier for inference.
    """
    
    def __init__(self, 
                 model_path: str,
                 scaler_path: str,
                 encoder_path: str,
                 device: str = 'auto'):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            encoder_path: Path to label encoder
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        print("Loading model and preprocessors...")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"✓ Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load preprocessor first to get model config
        self.preprocessor = DNSDataPreprocessor()
        self.preprocessor.load(scaler_path, encoder_path)
        print(f"✓ Preprocessor loaded")
        
        # Get model configuration from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            # Use default config if not found
            model_config = {}
        
        # Determine input dimension from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Find embedding layer weight to determine input dim
            for key in state_dict.keys():
                if 'embedding.weight' in key:
                    input_dim = state_dict[key].shape[1]
                    break
            else:
                # Default fallback
                input_dim = 50
        else:
            input_dim = 50
        
        # Build model
        num_classes = self.preprocessor.get_num_classes()
        self.model = build_cnn_lstm_model(input_dim, num_classes, model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is the state dict itself
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded from: {model_path}")
        
        # Initialize feature extractor
        self.feature_extractor = DNSFeatureExtractor()
        print(f"✓ Feature extractor initialized")
        
        # Get class names
        self.class_names = self.preprocessor.get_class_names()
        print(f"✓ Classes: {self.class_names}")
    
    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict class for a single DNS record.
        
        Args:
            record: DNS record dictionary
            
        Returns:
            Prediction result dictionary
        """
        # Extract features
        features = self.feature_extractor.extract_features(record)
        features = features.reshape(1, -1)
        
        # Preprocess
        features_scaled = self.preprocessor.transform(features)
        
        # Prepare for CNN
        features_cnn = prepare_data_for_cnn(features_scaled)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_cnn).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Prepare result
        result = {
            'domain': record.get('domain_name', 'unknown'),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }
        }
        
        return result
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict classes for multiple DNS records.
        
        Args:
            records: List of DNS record dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        # Extract features
        features = self.feature_extractor.extract_features_batch(records)
        
        # Preprocess
        features_scaled = self.preprocessor.transform(features)
        
        # Prepare for CNN
        features_cnn = prepare_data_for_cnn(features_scaled)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_cnn).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
        
        predicted_classes_idx = np.argmax(probabilities, axis=1)
        
        # Prepare results
        results = []
        for i, record in enumerate(records):
            predicted_class = self.class_names[predicted_classes_idx[i]]
            confidence = probabilities[i, predicted_classes_idx[i]]
            
            result = {
                'domain': record.get('domain_name', 'unknown'),
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, probabilities[i])
                }
            }
            results.append(result)
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str = None):
        """
        Predict classes for records in a JSON file.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (optional)
        """
        print(f"\nLoading records from: {input_file}")
        
        with open(input_file, 'r') as f:
            records = json.load(f)
        
        if not isinstance(records, list):
            records = [records]
        
        print(f"Processing {len(records)} records...")
        
        # Predict
        results = self.predict_batch(records)
        
        # Print results
        print("\nPrediction Results:")
        print("=" * 80)
        for result in results:
            print(f"\nDomain: {result['domain']}")
            print(f"  Predicted Class: {result['predicted_class'].upper()}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"    {class_name}: {prob:.4f}")
        print("=" * 80)
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results


def main(args):
    """Main prediction pipeline."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "MALICIOUS DOMAIN DETECTION - PREDICTION (PyTorch)")
    print("=" * 80 + "\n")
    
    # Initialize classifier
    classifier = DomainClassifier(
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder,
        device=args.device
    )
    
    # Predict from file or interactive mode
    if args.input:
        results = classifier.predict_from_file(args.input, args.output)
    else:
        print("\nInteractive mode: Enter domain information (JSON format)")
        print("Example: {\"domain_name\": \"example.com\", \"dns\": {...}}")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Enter DNS record (JSON): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Parse JSON
                record = json.loads(user_input)
                
                # Predict
                result = classifier.predict_single(record)
                
                # Print result
                print(f"\n--- Prediction ---")
                print(f"Domain: {result['domain']}")
                print(f"Predicted Class: {result['predicted_class'].upper()}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name}: {prob:.4f}")
                print()
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON format. Please try again.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETED")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict malicious domains (PyTorch)')
    
    parser.add_argument(
        '--model',
        type=str,
        default='saved_models/best_model.pt',
        help='Path to trained model file (.pt)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='processed_data/scaler.pkl',
        help='Path to feature scaler file'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='processed_data/label_encoder.pkl',
        help='Path to label encoder file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input JSON file with DNS records'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output JSON file for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = main(args)
        exit(exit_code)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
