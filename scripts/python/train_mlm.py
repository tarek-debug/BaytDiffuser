#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import train_masked_language_model

# Define paths
processed_data_path = '../data/processed/processed_taweel_data.csv'
mlm_output_dir = '../models/mlm'

os.makedirs(mlm_output_dir, exist_ok=True)

# Load processed data and subset for testing
print("Loading processed data for MLM training...")
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

# Use the texts from the subset for training MLM
texts = train_subset['text'].tolist()

# Train Masked Language Model (MLM)
print("Training Masked Language Model (MLM)...")
try:
    mlm_model, mlm_tokenizer, history = train_masked_language_model(
        texts=texts,
        model_name="aubmindlab/bert-base-arabertv2",
        max_length=128,
        epochs=3,
        batch_size=2,
        output_dir=mlm_output_dir,
        device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != 'nt' else 'cpu'
    )
    print("MLM training complete.")
except Exception as e:
    print(f"Error during MLM training: {e}")
    sys.exit(1)
