"""
Training script for Bach chorale generation model.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import BachChoraleLoader
from src.data.augmentation import BachChoraleAugmenter
from src.data.models.lstm_model import BachChoraleLSTM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Bach chorale generation model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing train, validation, and test data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to directory to save model and results')
    parser.add_argument('--augmentation_version', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Data augmentation version (1-4)')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Length of input sequences')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Dimension of note embeddings')
    parser.add_argument('--lstm_units', type=int, default=256,
                        help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs with no improvement after which to stop')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='Force use of GPU if available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training for faster GPU training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"bach_lstm_v{args.augmentation_version}_{timestamp}"
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure GPU if requested
    if args.use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU configuration successful. Found {len(gpus)} GPU(s).")
                
                # Enable mixed precision training if requested
                if args.mixed_precision:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    print("Mixed precision training enabled.")
                
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}")
        else:
            print("No GPU found. Using CPU for computation.")
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    loader = BachChoraleLoader(args.data_dir)
    train_chorales = loader.load_dataset('train')
    val_chorales = loader.load_dataset('val')
    
    # Augment data
    print(f"Applying augmentation version {args.augmentation_version}...")
    augmenter = BachChoraleAugmenter()
    augmented_train = augmenter.get_augmented_dataset(train_chorales, version=args.augmentation_version)
    augmented_val = augmenter.get_augmented_dataset(val_chorales, version=args.augmentation_version)
    
    print(f"Original training set: {len(train_chorales)} chorales")
    print(f"Augmented training set: {len(augmented_train)} chorales")
    
    # Prepare sequences
    print(f"Preparing sequences with length {args.sequence_length}...")
    X_train, y_train = loader.prepare_sequences(augmented_train, args.sequence_length)
    X_val, y_val = loader.prepare_sequences(augmented_val, args.sequence_length)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    
    # Build model
    print("Building model...")
    model = BachChoraleLSTM(
        input_shape=(args.sequence_length, 4),  # 4 voices
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy (average across all voices)
    plt.subplot(1, 2, 2)
    accuracy_keys = [k for k in history.keys() if 'accuracy' in k and 'val' not in k]
    val_accuracy_keys = [k for k in history.keys() if 'accuracy' in k and 'val' in k]
    
    avg_acc = np.mean([history[k] for k in accuracy_keys], axis=0)
    avg_val_acc = np.mean([history[k] for k in val_accuracy_keys], axis=0)
    
    plt.plot(avg_acc, label='Training Accuracy')
    plt.plot(avg_val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    print(f"Training complete. Model and results saved to {output_dir}")


if __name__ == '__main__':
    main()