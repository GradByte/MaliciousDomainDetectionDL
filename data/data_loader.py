"""
Data loader for efficiently streaming and parsing large JSON files.
Uses ijson for memory-efficient streaming of multi-GB JSON files.
"""

import ijson
import numpy as np
from tqdm import tqdm
import json
import os
from typing import Iterator, Dict, Any, List, Optional


class DNSDataLoader:
    """
    Efficient data loader for large DNS JSON files.
    Uses streaming to handle files that don't fit in memory.
    """
    
    def __init__(self, data_dir: str = "Zenodo"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the JSON files
        """
        self.data_dir = data_dir
        self.file_paths = {
            'phishing': os.path.join(data_dir, 'phishing.json'),
            'malware': os.path.join(data_dir, 'malware.json'),
            'benign_umbrella': os.path.join(data_dir, 'benign_umbrella.json'),
            'benign_cesnet': os.path.join(data_dir, 'benign_cesnet.json')
        }
    
    def stream_json_array(self, file_path: str, max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream JSON array items one by one without loading entire file.
        
        Args:
            file_path: Path to JSON file
            max_records: Maximum number of records to stream (None for all)
            
        Yields:
            Individual JSON objects from the array
        """
        count = 0
        with open(file_path, 'rb') as f:
            # Parse items in the JSON array
            parser = ijson.items(f, 'item')
            for item in parser:
                yield item
                count += 1
                if max_records and count >= max_records:
                    break
    
    def count_records(self, category: str) -> int:
        """
        Count total records in a category (approximation using file size).
        
        Args:
            category: One of 'phishing', 'malware', 'benign_umbrella', 'benign_cesnet'
            
        Returns:
            Approximate record count
        """
        file_path = self.file_paths.get(category)
        if not file_path or not os.path.exists(file_path):
            return 0
        
        # Fast counting using sampling
        sample_size = 100
        total_chars = 0
        record_count = 0
        
        for i, record in enumerate(self.stream_json_array(file_path, max_records=sample_size)):
            # Estimate size by converting to string (avoid JSON serialization issues)
            record_str = str(record)
            total_chars += len(record_str)
            record_count += 1
            
            if i >= sample_size - 1:
                break
        
        if record_count == 0:
            return 0
        
        # Estimate average record size
        avg_record_size = total_chars / record_count
        
        # Get file size and estimate total records
        file_size = os.path.getsize(file_path)
        estimated_count = int(file_size / avg_record_size)
        
        return estimated_count
    
    def load_balanced_sample(self, 
                            samples_per_category: Dict[str, int],
                            random_seed: int = 42) -> tuple:
        """
        Load a balanced sample from all categories.
        
        Args:
            samples_per_category: Dict mapping category to number of samples
                                 e.g., {'phishing': 100000, 'malware': 100000, 
                                       'benign_umbrella': 50000, 'benign_cesnet': 50000}
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (records_list, labels_list)
        """
        np.random.seed(random_seed)
        
        all_records = []
        all_labels = []
        
        # Map categories to labels
        label_map = {
            'phishing': 'phishing',
            'malware': 'malware',
            'benign_umbrella': 'benign',
            'benign_cesnet': 'benign'
        }
        
        for category, n_samples in samples_per_category.items():
            if n_samples <= 0:
                continue
                
            print(f"\nLoading {n_samples:,} samples from {category}...")
            
            file_path = self.file_paths.get(category)
            if not file_path or not os.path.exists(file_path):
                print(f"Warning: File not found for {category}")
                continue
            
            # Estimate total records
            total_records = self.count_records(category)
            print(f"Estimated total records in {category}: {total_records:,}")
            
            # Calculate sampling rate
            if n_samples >= total_records:
                # Take all records
                sampling_rate = 1.0
            else:
                sampling_rate = n_samples / total_records
            
            # Stream and sample records
            records = []
            for record in tqdm(self.stream_json_array(file_path), 
                             desc=f"Sampling {category}",
                             total=total_records):
                
                # Reservoir sampling or random sampling
                if np.random.random() < sampling_rate:
                    records.append(record)
                    
                    if len(records) >= n_samples:
                        break
            
            print(f"Loaded {len(records):,} records from {category}")
            
            # Add to results
            all_records.extend(records)
            all_labels.extend([label_map[category]] * len(records))
        
        return all_records, all_labels
    
    def load_sample_for_exploration(self, n_samples: int = 1000) -> Dict[str, List[Dict]]:
        """
        Load a small sample from each category for exploration.
        
        Args:
            n_samples: Number of samples per category
            
        Returns:
            Dictionary mapping category to list of records
        """
        samples = {}
        
        for category, file_path in self.file_paths.items():
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {category}")
                continue
            
            print(f"Loading {n_samples} samples from {category}...")
            records = []
            
            for i, record in enumerate(self.stream_json_array(file_path, max_records=n_samples)):
                records.append(record)
                if i >= n_samples - 1:
                    break
            
            samples[category] = records
            print(f"Loaded {len(records)} records from {category}")
        
        return samples
    
    def get_dataset_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with record counts per category
        """
        stats = {}
        
        print("Calculating dataset statistics...")
        for category in self.file_paths.keys():
            count = self.count_records(category)
            stats[category] = count
            print(f"{category}: ~{count:,} records")
        
        # Aggregate benign counts
        stats['total_benign'] = stats.get('benign_umbrella', 0) + stats.get('benign_cesnet', 0)
        stats['total_malicious'] = stats.get('phishing', 0) + stats.get('malware', 0)
        stats['total'] = stats['total_benign'] + stats['total_malicious']
        
        return stats


def test_data_loader():
    """Test the data loader functionality."""
    print("Testing DNS Data Loader")
    print("=" * 50)
    
    loader = DNSDataLoader()
    
    # Test 1: Get dataset statistics
    print("\n1. Dataset Statistics:")
    stats = loader.get_dataset_statistics()
    print(f"\nTotal records: ~{stats['total']:,}")
    print(f"Benign: ~{stats['total_benign']:,}")
    print(f"Malicious: ~{stats['total_malicious']:,}")
    
    # Test 2: Load small sample for exploration
    print("\n2. Loading exploration sample...")
    samples = loader.load_sample_for_exploration(n_samples=10)
    
    for category, records in samples.items():
        if records:
            print(f"\n{category}: {len(records)} records")
            print(f"Sample domain: {records[0].get('domain_name', 'N/A')}")
            print(f"Keys in record: {list(records[0].keys())}")


if __name__ == "__main__":
    test_data_loader()

