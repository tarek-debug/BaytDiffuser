# =============================================================================
# Diffusion Model Components
# =============================================================================

import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LayerNormalization, Dense, MultiHeadAttention, Add, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import Sequential

def create_diffusion_model(input_shape, model_params):
    """
    Creates a Transformer-based diffusion model for text generation with
    LayerNormalization layers set to 'float32' to ensure compatibility
    with mixed precision training.
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, encoding_dim).
        model_params (dict): Parameters for model architecture.
    
    Returns:
        tf.keras.Model: Compiled diffusion model.
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # LayerNormalization with dtype='float32'
    x = LayerNormalization(dtype='float32', name='layer_normalization')(inputs)
    
    # Transformer Blocks
    for i in range(model_params.get('num_transformer_blocks', 2)):
        # Multi-head self-attention
        attention = MultiHeadAttention(
            num_heads=model_params.get('num_heads', 4),
            key_dim=model_params.get('key_dim', 64),
            name=f'multi_head_attention_{i}'
        )(x, x)
        
        # Add & Normalize with dtype='float32'
        x = Add(name=f'add_attention_{i}')([x, attention])
        x = LayerNormalization(dtype='float32', name=f'layer_normalization_{i}_post_attention')(x)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(model_params.get('ffn_units', 256), activation='relu', name=f'dense_ffn_{i}_1'),
            Dense(input_shape[-1], name=f'dense_ffn_{i}_2')
        ], name=f'ffn_{i}')
        
        ffn_output = ffn(x)
        
        # Add & Normalize with dtype='float32'
        x = Add(name=f'add_ffn_{i}')([x, ffn_output])
        x = LayerNormalization(dtype='float32', name=f'layer_normalization_{i}_post_ffn')(x)
    
    # Output layer with Activation set to 'float32'
    outputs = Dense(
        input_shape[-1],
        activation=None,
        name='output_dense'
    )(x)
    
    # Ensure the output is in 'float32'
    outputs = Activation('linear', dtype='float32', name='output_layer')(outputs)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name='DiffusionModel')
    
    # Compile the model with 'adam' optimizer and appropriate loss and metrics
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Display the model summary
    model.summary()
    
    return model

def train_diffusion_model(model, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, output_dir):
    """
    Trains the diffusion model with specified parameters and callbacks.
    
    Args:
        model (tf.keras.Model): The diffusion model to train.
        X_train (np.ndarray or tf.Tensor): Training input data.
        Y_train (np.ndarray or tf.Tensor): Training target data.
        X_valid (np.ndarray or tf.Tensor): Validation input data.
        Y_valid (np.ndarray or tf.Tensor): Validation target data.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        output_dir (str): Directory to save model checkpoints and logs.
    
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure data is in float32
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float32)
    Y_valid = tf.convert_to_tensor(Y_valid, dtype=tf.float32)
    
    # Define callbacks
    checkpoint_path = os.path.join(
        output_dir, 
        "diffusion_model_epoch_{epoch:02d}_val_mae_{val_mae:.4f}.h5"
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_mae', 
        verbose=1,
        save_best_only=True, 
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_mae', 
        patience=5, 
        verbose=1, 
        mode='min',
        restore_best_weights=True
    )
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs"))
    
    callbacks_list = [checkpoint, early_stopping, tensorboard]
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the final model in 'float32' to ensure compatibility
    final_model_path = os.path.join(output_dir, "diffusion_model_final.h5")
    model.save(final_model_path, include_optimizer=False)
    print(f"Final diffusion model saved to {final_model_path}.")
    
    return history

def load_diffusion_model(model_path):
    """
    Loads a trained diffusion model from the specified path.
    
    Args:
        model_path (str): Path to the saved diffusion model (.h5 file).
    
    Returns:
        tf.keras.Model: Loaded diffusion model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Diffusion model not found at {model_path}")
    
    # Load the model with custom_objects if any custom layers or functions are used
    model = load_model(model_path, compile=True)
    print(f"Loaded diffusion model from {model_path}")
    return model

'''

def create_diffusion_model(input_shape, model_params):
    """
    Creates a Transformer-based diffusion model for text generation.
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, encoding_dim).
        model_params (dict): Parameters for model architecture.
    
    Returns:
        tf.keras.Model: Compiled diffusion model.
    """
    # Example Transformer-based architecture for diffusion
    inputs = Input(shape=input_shape, name='input_layer')
    x = LayerNormalization()(inputs)
    
    for _ in range(model_params.get('num_transformer_blocks', 2)):
        # Multi-head self-attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=model_params.get('num_heads', 4),
            key_dim=model_params.get('key_dim', 64)
        )(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = LayerNormalization()(x)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(model_params.get('ffn_units', 256), activation='relu'),
            Dense(input_shape[-1])
        ])
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = LayerNormalization()(x)
    
    # Output layer
    outputs = Dense(input_shape[-1], activation='linear', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='DiffusionModel')
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.summary()
    return model

def train_diffusion_model(model, train_data, valid_data, epochs, batch_size, output_dir):
    """
    Trains the diffusion model.
    
    Args:
        model (tf.keras.Model): The diffusion model to train.
        train_data (list of dict): Training data entries.
        valid_data (list of dict): Validation data entries.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        output_dir (str): Directory to save model checkpoints.
    
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare training data
    X_train = np.array([entry['text'] for entry in train_data])
    Y_train = X_train.copy()  # For diffusion models, target is often the clean data
    
    # Prepare validation data
    X_valid = np.array([entry['text'] for entry in valid_data])
    Y_valid = X_valid.copy()
    
    # Define callbacks
    checkpoint_path = os.path.join(output_dir, "diffusion_model_epoch_{epoch:02d}_val_mae_{val_mae:.4f}.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mae', verbose=1,
                                 save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_mae', patience=5, verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs"))
    
    callbacks_list = [checkpoint, early_stopping, tensorboard]
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "diffusion_model_final.h5")
    model.save(final_model_path)
    print(f"Final diffusion model saved to {final_model_path}.")
    
    return history

def load_diffusion_model(model_path):
    """
    Loads a trained diffusion model from the specified path.
    
    Args:
        model_path (str): Path to the saved diffusion model (.h5 file).
    
    Returns:
        tf.keras.Model: Loaded diffusion model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Diffusion model not found at {model_path}")
    model = load_model(model_path)
    print(f"Loaded diffusion model from {model_path}")
    return model


'''