#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (
    create_diffusion_model_pytorch,
    train_diffusion_with_gpt2_decoder,
    AutoTokenizer,
    ArabertPreprocessor
)

# Define paths
processed_data_path = '../data/processed/processed_taweel_data.csv'
diffusion_output_dir = '../models/diffusion'

os.makedirs(diffusion_output_dir, exist_ok=True)

# Load processed data and subset for testing
print("Loading processed data for Diffusion Model training...")
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
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

try:
    preprocessor = ArabertPreprocessor(model_name='aubmindlab/arabertv2')
except Exception as e:
    print(f"Error initializing preprocessor: {e}")
    sys.exit(1)

# Create Diffusion model
max_bayt_len = 128
encoding_dim = 8
diffusion_model_params = {
    'num_transformer_blocks': 4,
    'num_heads': 8,
    'key_dim': 64,
    'ffn_units': 512
}
input_shape = (max_bayt_len, encoding_dim)
diffusion_model = create_diffusion_model_pytorch(input_shape, diffusion_model_params)

# Train Diffusion Model with GPT2 Decoder
print("Training Diffusion Model with GPT2 Decoder...")
try:
    trained_diffusion, history = train_diffusion_with_gpt2_decoder(
        df_classical=train_subset,
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=128,
        max_bayt_len=max_bayt_len,
        encoding_dim=encoding_dim,
        epochs=10,
        batch_size=8,
        output_dir=diffusion_output_dir,
        learning_rate=1e-4,
        patience=3,
        device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != 'nt' else 'cpu'
    )
    final_diffusion_path = os.path.join(diffusion_output_dir, 'final_diffusion_model_with_decoder.pt')
    import torch
    torch.save(trained_diffusion.state_dict(), final_diffusion_path)
    print(f"Diffusion model saved to '{final_diffusion_path}'.")
except Exception as e:
    print(f"Error during Diffusion training: {e}")
    sys.exit(1)
