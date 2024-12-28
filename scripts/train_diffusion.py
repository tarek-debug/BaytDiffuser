"""
train_diffusion.py

Trains a diffusion model to iteratively refine text embeddings.
The model takes noisy text representations from the transformer output 
and refines them to produce classical Arabic poetry.

Usage:
    python train_diffusion.py --train_data ../data/processed
"""

import os
import argparse
from utils import load_processed_data, create_diffusion_model, train_diffusion_model

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed training data.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="../models/diffusion",
                        help="Directory to save trained diffusion model checkpoints.")
    args = parser.parse_args()

    # Load data
    train_dataset = load_processed_data(args.train_data)

    # Create a diffusion model
    diffusion_model = create_diffusion_model()

    # Train the diffusion model
    train_diffusion_model(diffusion_model, train_dataset,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          output_dir=args.output_dir)

    print(f"Diffusion model training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
