"""
Data augmentation for Bach chorales dataset.
Implements various augmentation strategies including transposition and tempo changes.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class BachChoraleAugmenter:
    """
    Augmenter for Bach chorales dataset.
    Implements various augmentation strategies including transposition and tempo changes.
    """
    
    def __init__(self):
        """Initialize the augmenter."""
        # Common key transpositions in semitones
        self.key_transpositions = {
            'C_major': 0,  # C major / A minor (no transposition)
            'G_major': 7,  # G major / E minor (up a fifth)
            'F_major': 5,  # F major / D minor (up a fourth)
            # All 12 keys (0-11 semitones)
            'all_keys': list(range(12))
        }
    
    def transpose_chorale(self, chorale: np.ndarray, semitones: int) -> np.ndarray:
        """
        Transpose a chorale by a given number of semitones.
        
        Args:
            chorale: Numpy array of shape (time_steps, 4) with MIDI note values
            semitones: Number of semitones to transpose by (positive = up, negative = down)
            
        Returns:
            Transposed chorale
        """
        transposed = chorale.copy().astype(float)
        
        # Only transpose non-zero notes and ensure we're working with numeric values
        mask = (transposed > 0) & (~np.isnan(transposed))
        transposed[mask] = transposed[mask] + semitones
        
        # Ensure notes are within MIDI range (21-108)
        too_low_mask = (transposed < 21) & (transposed > 0) & (~np.isnan(transposed))
        transposed[too_low_mask] = transposed[too_low_mask] + 12
        
        too_high_mask = (transposed > 108) & (~np.isnan(transposed))
        transposed[too_high_mask] = transposed[too_high_mask] - 12
        
        return transposed
    
    def change_tempo(self, chorale: np.ndarray, factor: float) -> np.ndarray:
        """
        Change the tempo of a chorale by a given factor.
        
        Args:
            chorale: Numpy array of shape (time_steps, 4) with MIDI note values
            factor: Factor to change tempo by (0.5 = half speed, 2.0 = double speed)
            
        Returns:
            Tempo-adjusted chorale
        """
        if factor == 1.0:
            return chorale
        
        time_steps, voices = chorale.shape
        
        if factor < 1.0:  # Slower tempo (expand)
            new_time_steps = int(time_steps / factor)
            tempo_adjusted = np.zeros((new_time_steps, voices))
            
            for i in range(time_steps):
                idx = int(i / factor)
                tempo_adjusted[idx] = chorale[i]
        else:  # Faster tempo (compress)
            new_time_steps = int(time_steps * factor)
            tempo_adjusted = np.zeros((new_time_steps, voices))
            
            for i in range(new_time_steps):
                idx = int(i * factor)
                if idx < time_steps:
                    tempo_adjusted[i] = chorale[idx]
        
        return tempo_adjusted
    
    def normalize_to_c_major(self, chorales: List[np.ndarray]) -> List[np.ndarray]:
        """
        Version 2: Transpose all chorales to C major/A minor.
        
        Args:
            chorales: List of numpy arrays representing chorales
            
        Returns:
            List of transposed chorales
        """
        normalized_chorales = []
        
        for chorale in chorales:
            # A simple heuristic - look at first chord to determine key
            # This is a simplification; actual key detection would be more complex
            first_chord = chorale[0]
            non_zero_notes = first_chord[first_chord > 0]
            
            if len(non_zero_notes) == 0:
                # If first chord has no notes, use the first chord that does
                for i in range(1, min(10, chorale.shape[0])):
                    potential_notes = chorale[i][chorale[i] > 0]
                    if len(potential_notes) > 0:
                        non_zero_notes = potential_notes
                        break
            
            if len(non_zero_notes) == 0:
                # If still no notes found, just add the original
                normalized_chorales.append(chorale)
                continue
            
            # Determine approximate key by looking at notes mod 12
            note_classes = non_zero_notes % 12
            
            # Count occurrences of each pitch class
            unique, counts = np.unique(note_classes, return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            # Calculate semitones to transpose to C (pitch class 0)
            semitones_to_c = (12 - most_common) % 12
            
            # Transpose to C
            transposed = self.transpose_chorale(chorale, semitones_to_c)
            normalized_chorales.append(transposed)
        
        return normalized_chorales
    
    def augment_with_common_keys(self, chorales: List[np.ndarray]) -> List[np.ndarray]:
        """
        Version 3: Transpose to C, G, and F majors with tempo variations.
        
        Args:
            chorales: List of numpy arrays representing chorales
            
        Returns:
            List of augmented chorales
        """
        # First normalize to C major
        c_major_chorales = self.normalize_to_c_major(chorales)
        augmented_chorales = c_major_chorales.copy()
        
        # Add transpositions to G major and F major
        for chorale in c_major_chorales:
            # Transpose to G major (up a fifth)
            g_major = self.transpose_chorale(chorale, self.key_transpositions['G_major'])
            augmented_chorales.append(g_major)
            
            # Transpose to F major (up a fourth)
            f_major = self.transpose_chorale(chorale, self.key_transpositions['F_major'])
            augmented_chorales.append(f_major)
        
        # Add tempo variations (half and double speed)
        tempo_chorales = []
        for chorale in augmented_chorales:
            # Half speed
            half_speed = self.change_tempo(chorale, 0.5)
            tempo_chorales.append(half_speed)
            
            # Double speed
            double_speed = self.change_tempo(chorale, 2.0)
            tempo_chorales.append(double_speed)
        
        augmented_chorales.extend(tempo_chorales)
        
        return augmented_chorales
    
    def augment_with_all_keys(self, chorales: List[np.ndarray]) -> List[np.ndarray]:
        """
        Version 4: Transpose to all 12 keys with tempo variations (half, normal, double).
        
        Args:
            chorales: List of numpy arrays representing chorales
            
        Returns:
            List of augmented chorales
        """
        # First normalize to C major
        c_major_chorales = self.normalize_to_c_major(chorales)
        augmented_chorales = []
        
        # Transpose to all 12 keys with different tempos
        for chorale in c_major_chorales:
            for semitones in self.key_transpositions['all_keys']:
                # Normal tempo
                transposed = self.transpose_chorale(chorale, semitones)
                augmented_chorales.append(transposed)
                
                # Half tempo
                half_speed = self.change_tempo(transposed, 0.5)
                augmented_chorales.append(half_speed)
                
                # Double tempo
                double_speed = self.change_tempo(transposed, 2.0)
                augmented_chorales.append(double_speed)
        
        return augmented_chorales
    
    def get_augmented_dataset(self, chorales: List[np.ndarray], version: int = 1) -> List[np.ndarray]:
        """
        Get the augmented dataset based on the specified version.
        
        Args:
            chorales: List of numpy arrays representing chorales
            version: Augmentation version (1-4)
            
        Returns:
            List of augmented chorales
        """
        if version == 1:
            # Version 1: Original dataset (no augmentation)
            return chorales
        elif version == 2:
            # Version 2: All transposed to C major/A minor
            return self.normalize_to_c_major(chorales)
        elif version == 3:
            # Version 3: Version 2 + copies in G and F major + tempo variations
            return self.augment_with_common_keys(chorales)
        elif version == 4:
            # Version 4: All 12 keys
            return self.augment_with_all_keys(chorales)
        else:
            raise ValueError(f"Invalid augmentation version: {version}. Must be 1-4.")