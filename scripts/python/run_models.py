#!/usr/bin/env python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Import necessary functions from utils.py
from utils import (
    generate_classical_poem_with_thepoet,
    AutoTokenizer,
    ArabertPreprocessor,
    create_diffusion_model_pytorch,
    train_aragpt2_for_classical_style  # for loading if needed
)
import torch

# Define paths to saved models and processed data
processed_data_path = '../data/processed/processed_taweel_data.csv'
transformer_model_dir = '../models/transformers'
diffusion_model_path = '../models/diffusion/final_diffusion_model_with_decoder.pt'
dae_model_path = '../models/dae/denoising_autoencoder_final.pt'
mlm_model_dir = '../models/mlm/masked_lm_model'
# (RL model is used during training for reinforcement; here we focus on generation)

# Load a small subset for testing the inference pipeline (not used directly for generation)
print("Loading processed data for inference testing...")
try:
    processed_df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
    _, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    test_subset = test_df.sample(n=10, random_state=42)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Load the Transformer (AraGPT2) model and tokenizer
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_dir)
    transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_dir)
    print("Transformer model and tokenizer loaded.")
except Exception as e:
    print(f"Error loading Transformer model: {e}")
    sys.exit(1)

# Load the Diffusion model
try:
    max_bayt_len = 128
    encoding_dim = 8
    diffusion_model_params = {'num_transformer_blocks': 4, 'num_heads': 8, 'key_dim': 64, 'ffn_units': 512}
    input_shape = (max_bayt_len, encoding_dim)
    diffusion_model = create_diffusion_model_pytorch(input_shape, diffusion_model_params)
    diffusion_model.load_state_dict(torch.load(diffusion_model_path, map_location='cpu'))
    print("Diffusion model loaded.")
except Exception as e:
    print(f"Error loading Diffusion model: {e}")
    diffusion_model = None

# Load the DAE model
try:
    dae_tokenizer = AutoTokenizer.from_pretrained(transformer_model_dir)  # using the same tokenizer
    # We assume the DAE architecture is defined in utils.py
    from utils import create_denoising_autoencoder
    vocab_size = dae_tokenizer.vocab_size
    dae_model = create_denoising_autoencoder(vocab_size, embedding_dim=128, latent_dim=256, max_length=128)
    dae_model.load_state_dict(torch.load(dae_model_path, map_location='cpu'))
    print("DAE model loaded.")
except Exception as e:
    print(f"Error loading DAE model: {e}")
    dae_model = None

# Load the MLM model
try:
    from transformers import AutoModelForMaskedLM
    mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_model_dir)
    mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_model_dir)
    print("MLM model loaded.")
except Exception as e:
    print(f"Error loading MLM model: {e}")
    mlm_model = None

# Initialize ThePoet pipeline
try:
    from transformers import pipeline
    poet_pipeline = pipeline(
        "text-generation",
        model="mabaji/thepoet",
        tokenizer="mabaji/thepoet",
        device=0 if torch.cuda.is_available() else -1
    )
    print("ThePoet pipeline created.")
except Exception as e:
    print(f"Error creating ThePoet pipeline: {e}")
    sys.exit(1)

# Initialize preprocessor for Transformer
try:
    preprocessor = ArabertPreprocessor(model_name='aubmindlab/arabertv2')
except Exception as e:
    print(f"Error initializing ArabertPreprocessor: {e}")
    sys.exit(1)

# Define a modern prompt for generating the final poem
modern_prompt = "يا جمال الزمان ويا نور الأمل"
print("Generating final classical poem using all models...")

try:
    final_poem = generate_classical_poem_with_thepoet(
        modern_prompt=modern_prompt,
        poet_pipeline=poet_pipeline,
        transformer_model=transformer_model,
        transformer_tokenizer=transformer_tokenizer,
        diffusion_model=diffusion_model,  # may be None if not loaded
        dae_model=dae_model,              # may be None if not loaded
        mlm_model=mlm_model,              # may be None if not loaded
        max_length=128,
        max_bayt_len=max_bayt_len,
        encoding_dim=encoding_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("\n==== Final Chained Poem ====")
    print(final_poem)
    print("================================")
except Exception as e:
    print(f"Error during poem generation: {e}")
