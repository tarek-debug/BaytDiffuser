#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import functions from utils.py
from utils import (
    train_aragpt2_for_classical_style,
    AutoTokenizer,
    ArabertPreprocessor
)

# Define paths
processed_data_path = '../data/processed/processed_taweel_data.csv'
transformer_output_dir = '../models/transformers'

# Create output directory if it doesn't exist
os.makedirs(transformer_output_dir, exist_ok=True)

# Load processed data and subset for testing
print("Loading processed data for Transformer training...")
try:
    processed_df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
    print(f"Data loaded with {len(processed_df)} records.")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Subset data for testing
subset = True
if subset:
    train_df, _ = train_test_split(processed_df, test_size=0.2, random_state=42)
    train_subset = train_df.sample(n=100, random_state=42)
else:
    train_subset = processed_df

print(f"Training on {len(train_subset)} records (subset).")

# Initialize tokenizer and preprocessor
transformer_model_name = "aubmindlab/aragpt2-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Assigned EOS token as PAD token.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

try:
    preprocessor = ArabertPreprocessor(model_name='aubmindlab/arabertv2')
    print("ArabertPreprocessor initialized.")
except Exception as e:
    print(f"Error initializing preprocessor: {e}")
    sys.exit(1)

# Train the AraGPT2 Transformer model for classical style
print("Training Transformer model (AraGPT2)...")
try:
    trained_model, history = train_aragpt2_for_classical_style(
        df_classical=train_subset,
        tokenizer=tokenizer,
        model=None,  # Let the function initialize the model via from_pretrained inside utils if needed
        preprocessor=preprocessor,
        max_length=128,
        epochs=10,
        batch_size=4,
        output_dir=transformer_output_dir,
        device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != 'nt' else 'cpu',
        freeze_layers=0,
        weight_decay=0.01,
        patience=3,
        max_grad_norm=1.0
    )
    print("Transformer training complete.")
except Exception as e:
    print(f"Error during Transformer training: {e}")
    sys.exit(1)

# Save the trained model and tokenizer
try:
    trained_model.save_pretrained(transformer_output_dir)
    tokenizer.save_pretrained(transformer_output_dir)
    print(f"Transformer model and tokenizer saved to '{transformer_output_dir}'.")
except Exception as e:
    print(f"Error saving Transformer model/tokenizer: {e}")
    sys.exit(1)

# Optionally plot training history
try:
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("AraGPT2 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Error plotting history: {e}")
