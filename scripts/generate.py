"""
Script to generate Bach chorales using a trained model.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_model import BachChoraleLSTM
from src.data.loader import BachChoraleLoader
from src.utils.midi_conversion import save_chorale_as_midi


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Bach chorales using a trained model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model file (.h5)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing train, validation, and test data')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Path to directory to save generated chorales')
    parser.add_argument('--num_chorales', type=int, default=5,
                        help='Number of chorales to generate')
    parser.add_argument('--chorale_length', type=int, default=100,
                        help='Length of generated chorales (time steps)')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Length of seed sequence (must match training)')
    parser.add_argument('--use_seed', action='store_true',
                        help='Use a real chorale segment as seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    
    return parser.parse_args()


def generate_with_temperature(model, seed, length, temperature=1.0):
    """
    Generate a sequence with temperature sampling.
    
    Args:
        model: Trained BachChoraleLSTM model
        seed: Seed sequence
        length: Length of sequence to generate
        temperature: Sampling temperature (1.0 = normal, <1.0 = more conservative, >1.0 = more random)
    
    Returns:
        Generated sequence
    """
    generated_sequence = seed.copy()
    
    for _ in range(length):
        # Get the last sequence_length time steps
        current_sequence = generated_sequence[-model.input_shape[0]:]
        current_sequence = current_sequence.reshape(1, *current_sequence.shape)
        
        # Get raw predictions (before argmax)
        raw_predictions = model.model.predict(current_sequence)
        
        # Apply temperature scaling
        next_notes = []
        for voice_pred in raw_predictions:
            # Scale logits by temperature
            scaled_logits = np.log(voice_pred[0] + 1e-10) / temperature
            # Convert back to probabilities
            scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
            # Sample from distribution
            next_note = np.random.choice(len(scaled_probs), p=scaled_probs)
            next_notes.append(next_note)
        
        # Add to generated sequence
        generated_sequence = np.vstack([generated_sequence, next_notes])
    
    return generated_sequence


def main():
    """Main generation function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"generated_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = BachChoraleLSTM.load(args.model_path)
    
    # Get seed sequences
    if args.use_seed:
        print("Using real chorale segments as seeds...")
        loader = BachChoraleLoader(args.data_dir)
        test_chorales = loader.load_dataset('test')
        
        for i in range(args.num_chorales):
            # Select a random chorale and starting point
            chorale_idx = np.random.randint(0, len(test_chorales))
            chorale = test_chorales[chorale_idx]
            
            # Ensure the chorale is long enough
            if chorale.shape[0] <= args.sequence_length:
                continue
            
            # Select a random starting point
            start_idx = np.random.randint(0, chorale.shape[0] - args.sequence_length)
            seed_sequence = chorale[start_idx:start_idx+args.sequence_length]
            
            # Generate sequence
            print(f"Generating chorale {i+1}/{args.num_chorales}...")
            generated = generate_with_temperature(
                model, seed_sequence, args.chorale_length, args.temperature
            )
            
            # Save as CSV
            output_csv = os.path.join(output_dir, f"generated_chorale_{i+1}.csv")
            pd.DataFrame(generated).to_csv(output_csv, index=False, header=False)
            
            # Save as MIDI
            output_midi = os.path.join(output_dir, f"generated_chorale_{i+1}.mid")
            save_chorale_as_midi(generated, output_midi)
            
            # Visualize
            plt.figure(figsize=(15, 8))
            colors = ['r', 'g', 'b', 'purple']
            voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
            
            for j in range(4):
                voice_data = generated[:, j]
                time_steps = np.arange(len(voice_data))
                valid_indices = voice_data > 0
                valid_time_steps = time_steps[valid_indices]
                valid_notes = voice_data[valid_indices]
                plt.scatter(valid_time_steps, valid_notes, color=colors[j], alpha=0.6, label=voice_names[j])
            
            plt.axvline(x=args.sequence_length, color='k', linestyle='--', label='Seed End')
            plt.xlabel('Time Step')
            plt.ylabel('MIDI Note Value')
            plt.title(f'Generated Chorale {i+1}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"generated_chorale_{i+1}.png"))
            plt.close()
    else:
        print("Using random seeds...")
        # Use random seeds
        for i in range(args.num_chorales):
            # Create a random seed sequence
            # Note values are typically between 40 and 80 for the four voices
            seed_sequence = np.zeros((args.sequence_length, 4))
            for j in range(args.sequence_length):
                seed_sequence[j] = [
                    np.random.randint(60, 80),  # Soprano: higher
                    np.random.randint(50, 70),  # Alto: mid-high
                    np.random.randint(40, 60),  # Tenor: mid-low
                    np.random.randint(30, 50)   # Bass: lower
                ]
            
            # Generate sequence
            print(f"Generating chorale {i+1}/{args.num_chorales}...")
            generated = generate_with_temperature(
                model, seed_sequence, args.chorale_length, args.temperature
            )
            
            # Save as CSV
            output_csv = os.path.join(output_dir, f"generated_chorale_{i+1}.csv")
            pd.DataFrame(generated).to_csv(output_csv, index=False, header=False)
            
            # Save as MIDI
            output_midi = os.path.join(output_dir, f"generated_chorale_{i+1}.mid")
            save_chorale_as_midi(generated, output_midi)
            
            # Visualize
            plt.figure(figsize=(15, 8))
            colors = ['r', 'g', 'b', 'purple']
            voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
            
            for j in range(4):
                voice_data = generated[:, j]
                time_steps = np.arange(len(voice_data))
                valid_indices = voice_data > 0
                valid_time_steps = time_steps[valid_indices]
                valid_notes = voice_data[valid_indices]
                plt.scatter(valid_time_steps, valid_notes, color=colors[j], alpha=0.6, label=voice_names[j])
            
            plt.axvline(x=args.sequence_length, color='k', linestyle='--', label='Seed End')
            plt.xlabel('Time Step')
            plt.ylabel('MIDI Note Value')
            plt.title(f'Generated Chorale {i+1}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"generated_chorale_{i+1}.png"))
            plt.close()
    
    print(f"Generated {args.num_chorales} chorales and saved to {output_dir}")


if __name__ == '__main__':
    main()