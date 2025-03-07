"""
Data loader for Bach chorales dataset.
Handles reading CSV files and converting to appropriate format for model training.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from glob import glob


class BachChoraleLoader:
    """
    Loader for Bach chorales dataset in CSV format.
    Each CSV file contains a chorale with 4 voices (soprano, alto, tenor, bass).
    Each row represents a time step, and the 4 columns represent the 4 voices.
    Values are MIDI note numbers (0 = no note played).
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader with the directory containing the data.
        
        Args:
            data_dir: Path to directory containing train, validation, and test subdirectories
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'validation')
        self.test_dir = os.path.join(data_dir, 'test')
        
        # Verify that directories exist
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    def load_dataset(self, split: str = 'train') -> List[np.ndarray]:
        """
        Load all chorales from the specified split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            List of numpy arrays, where each array has shape (time_steps, 4)
        """
        if split == 'train':
            dir_path = self.train_dir
        elif split == 'val':
            dir_path = self.val_dir
        elif split == 'test':
            dir_path = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
        
        chorale_files = glob(os.path.join(dir_path, '*.csv'))
        chorales = []
        
        for file_path in chorale_files:
            try:
                # Read CSV file with explicit numeric conversion
                chorale_df = pd.read_csv(file_path, header=None)
                # Convert all columns to numeric type, coerce errors to NaN
                chorale = chorale_df.apply(pd.to_numeric, errors='coerce').values
                chorales.append(chorale)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(chorales)} chorales from {split} split")
        return chorales
    
    def validate_dataset(self, chorales: List[np.ndarray]) -> None:
        """
        Validate the dataset for common issues.
        
        Args:
            chorales: List of numpy arrays representing chorales
        """
        print(f"Validating {len(chorales)} chorales...")
        
        # Check for non-numeric values
        non_numeric_count = 0
        nan_count = 0
        for i, chorale in enumerate(chorales):
            # Check data type
            if not np.issubdtype(chorale.dtype, np.number):
                print(f"  Warning: Chorale {i} has non-numeric dtype: {chorale.dtype}")
                non_numeric_count += 1
            
            # Check for NaN values
            if np.isnan(chorale).any():
                print(f"  Warning: Chorale {i} contains NaN values")
                nan_count += 1
                
            # Check for string values
            try:
                chorale.astype(float)
            except (ValueError, TypeError):
                print(f"  Warning: Chorale {i} contains values that cannot be converted to float")
        
        print(f"  Found {non_numeric_count} chorales with non-numeric dtype")
        print(f"  Found {nan_count} chorales with NaN values")
        
        # Sample data check
        if chorales:
            sample_idx = np.random.randint(0, len(chorales))
            sample = chorales[sample_idx]
            print(f"\nSample chorale (index {sample_idx}):")
            print(f"  Shape: {sample.shape}")
            print(f"  Data type: {sample.dtype}")
            print(f"  First few rows:")
            try:
                print(sample[:5])
            except:
                print("  Error printing sample rows")
    
    def get_dataset_stats(self, chorales: List[np.ndarray]) -> Dict:
        """
        Compute statistics for the dataset.
        
        Args:
            chorales: List of numpy arrays representing chorales
            
        Returns:
            Dictionary with dataset statistics
        """
        total_time_steps = sum(chorale.shape[0] for chorale in chorales)
        min_length = min(chorale.shape[0] for chorale in chorales)
        max_length = max(chorale.shape[0] for chorale in chorales)
        avg_length = total_time_steps / len(chorales)
        
        # Get unique notes for each voice
        all_notes = np.concatenate([chorale.flatten() for chorale in chorales])
        
        # Ensure all values are numeric
        all_notes = pd.to_numeric(pd.Series(all_notes), errors='coerce').values
        
        # Filter out NaN values
        all_notes = all_notes[~np.isnan(all_notes)]
        
        unique_notes = np.unique(all_notes)
        
        # Handle the case where there might be no positive values
        positive_notes = all_notes[all_notes > 0]
        if len(positive_notes) > 0:
            note_range = (np.min(positive_notes), np.max(all_notes))
        else:
            note_range = (None, np.max(all_notes) if len(all_notes) > 0 else None)
        
        return {
            'num_chorales': len(chorales),
            'total_time_steps': total_time_steps,
            'min_length': min_length,
            'max_length': max_length,
            'avg_length': avg_length,
            'unique_notes': unique_notes,
            'note_range': note_range
        }
    
    def prepare_sequences(self, chorales: List[np.ndarray], 
                         sequence_length: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input-output sequences for training an LSTM model.
        
        Args:
            chorales: List of numpy arrays representing chorales
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target sequences
        """
        X, y = [], []
        
        for chorale in chorales:
            # Ensure the chorale is long enough
            if chorale.shape[0] <= sequence_length:
                continue
                
            # Ensure all values are numeric
            chorale_numeric = chorale.astype(float)
            
            # Check for and handle NaN values
            if np.isnan(chorale_numeric).any():
                # Fill NaN values with 0 (no note)
                chorale_numeric = np.nan_to_num(chorale_numeric, nan=0.0)
            
            # Create sequences
            for i in range(chorale_numeric.shape[0] - sequence_length):
                X.append(chorale_numeric[i:i+sequence_length])
                y.append(chorale_numeric[i+sequence_length])
        
        return np.array(X), np.array(y)