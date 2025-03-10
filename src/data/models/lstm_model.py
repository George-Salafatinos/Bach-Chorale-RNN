"""
LSTM model for Bach chorales generation.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Dict, Optional


class BachChoraleLSTM:
    """
    LSTM model for generating Bach chorales.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],  # (sequence_length, num_voices)
                 num_notes: int = 109,  # MIDI notes range from 0-108
                 embedding_dim: int = 32,
                 lstm_units: int = 256,
                 dropout_rate: float = 0.3):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, num_voices)
            num_notes: Number of possible note values
            embedding_dim: Dimension of note embeddings
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_notes = num_notes
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Input shape: (batch_size, sequence_length, num_voices)
        input_layer = layers.Input(shape=self.input_shape)
        
        # Reshape to handle each voice separately
        reshaped = layers.Reshape((self.input_shape[0] * self.input_shape[1],))(input_layer)
        
        # Embed the note values
        embedding = layers.Embedding(self.num_notes, self.embedding_dim)(reshaped)
        
        # Reshape back to sequence format
        reshaped_embedding = layers.Reshape((self.input_shape[0], self.input_shape[1] * self.embedding_dim))(embedding)
        
        # LSTM layers
        lstm1 = layers.LSTM(self.lstm_units, return_sequences=True)(reshaped_embedding)
        dropout1 = layers.Dropout(self.dropout_rate)(lstm1)
        
        lstm2 = layers.LSTM(self.lstm_units)(dropout1)
        dropout2 = layers.Dropout(self.dropout_rate)(lstm2)
        
        # Dense layers for each voice
        voice_outputs = []
        for i in range(self.input_shape[1]):
            voice_dense = layers.Dense(128, activation='relu')(dropout2)
            voice_output = layers.Dense(self.num_notes, activation='softmax', name=f'voice_{i}')(voice_dense)
            voice_outputs.append(voice_output)
        
        # Create model
        model = models.Model(inputs=input_layer, outputs=voice_outputs)
        
        # Compile model with unique metric names
        metrics_dict = {}
        for i in range(self.input_shape[1]):
            metrics_dict[f'voice_{i}'] = [tf.keras.metrics.SparseCategoricalAccuracy(name=f'voice_{i}_accuracy')]
            
        model.compile(
            optimizer='adam',
            loss=['sparse_categorical_crossentropy'] * self.input_shape[1],
            metrics=metrics_dict
        )
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 64,
              epochs: int = 50,
              callbacks: Optional[list] = None) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            X_val: Validation input sequences
            y_val: Validation target sequences
            batch_size: Batch size for training
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Split y into separate outputs for each voice
        y_train_voices = [y_train[:, i] for i in range(y_train.shape[1])]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_voices = [y_val[:, i] for i in range(y_val.shape[1])]
            validation_data = (X_val, y_val_voices)
        
        history = self.model.fit(
            X_train, 
            y_train_voices,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted next time step for each sequence
        """
        predictions = self.model.predict(X)
        
        # Convert to note indices
        predicted_notes = np.zeros((X.shape[0], len(predictions)))
        for i, voice_pred in enumerate(predictions):
            predicted_notes[:, i] = np.argmax(voice_pred, axis=1)
        
        return predicted_notes
    
    def generate_sequence(self, seed_sequence: np.ndarray, length: int = 100) -> np.ndarray:
        """
        Generate a new chorale sequence starting from a seed sequence.
        
        Args:
            seed_sequence: Initial sequence to start generation from
            length: Length of sequence to generate
            
        Returns:
            Generated sequence
        """
        generated_sequence = seed_sequence.copy()
        
        for _ in range(length):
            # Get the last sequence_length time steps
            current_sequence = generated_sequence[-self.input_shape[0]:]
            current_sequence = current_sequence.reshape(1, *current_sequence.shape)
            
            # Predict next time step
            next_step = self.predict(current_sequence)
            
            # Add to generated sequence
            generated_sequence = np.vstack([generated_sequence, next_step[0]])
        
        return generated_sequence
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BachChoraleLSTM':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BachChoraleLSTM model
        """
        model = models.load_model(filepath)
        
        # Extract model parameters
        input_shape = model.input_shape[1:]
        lstm_units = model.get_layer(index=3).units  # Assuming first LSTM layer is at index 3
        
        # Create instance
        instance = cls(input_shape=input_shape, lstm_units=lstm_units)
        instance.model = model
        
        return instance