#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import train_denoising_autoencoder, AutoTokenizer

# Define paths
processed_data_path = '../data/processed/processed_taweel_data.csv'
dae_output_dir = '../models/dae'

os.makedirs(dae_output_dir, exist_ok=True)

# Load processed data and subset for testing
print("Loading processed data for DAE training...")
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

# Use the texts from the subset for training the DAE
texts = train_subset['text'].tolist()

# Initialize tokenizer (using same transformer tokenizer)
dae_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
if dae_tokenizer.pad_token is None:
    dae_tokenizer.pad_token = dae_tokenizer.eos_token

# Train Denoising Autoencoder
print("Training Denoising Autoencoder (DAE)...")
try:
    dae_model, history = train_denoising_autoencoder(
        texts=texts,
        tokenizer=dae_tokenizer,
        max_length=128,
        epochs=5,
        batch_size=4,
        output_dir=dae_output_dir,
        device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != 'nt' else 'cpu'
    )
    print("DAE training complete.")
except Exception as e:
    print(f"Error during DAE training: {e}")
    sys.exit(1)
