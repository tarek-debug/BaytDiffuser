#!/usr/bin/env python
import os
import sys
from utils import train_poetic_rl

# Define output directory for RL model
rl_output_dir = '../models/rl'
os.makedirs(rl_output_dir, exist_ok=True)

# Train Poetic RL Model
print("Training Poetic RL Model...")
try:
    rl_model, rl_tokenizer = train_poetic_rl(
        model_name="aubmindlab/aragpt2-base",
        initial_prompt="يا ليل الصب متى غده",
        episodes=5,
        max_length=20,
        alpha=0.9,
        device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != 'nt' else 'cpu'
    )
    print("RL training complete.")
except Exception as e:
    print(f"Error during RL training: {e}")
    sys.exit(1)
