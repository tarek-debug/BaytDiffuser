"""
train_transformer.py

This script trains a transformer-based model (e.g. AraBERT, AraGPT2, or T5)
for transforming modern Arabic poetry into classical style.

Usage:
    python train_transformer.py --train_data ../data/processed --model_name AraGPT2
"""

import os
import argparse
from utils import load_processed_data, create_transformer_model, train_model

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer-based model.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed training data.")
    parser.add_argument("--model_name", type=str, default="AraGPT2",
                        help="Name of the transformer model to use.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="../models/transformers",
                        help="Directory to save trained model checkpoints.")
    args = parser.parse_args()

    # Load data
    train_dataset = load_processed_data(args.train_data)

    # Create or load transformer model
    model, tokenizer = create_transformer_model(args.model_name)

    # Train the model
    train_model(model, tokenizer, train_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir)

    print(f"Transformer training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
