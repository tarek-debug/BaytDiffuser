"""
train_transformer.py

This script trains a transformer-based model (e.g., AraBERT, AraGPT2, or T5)
to transform modern Arabic poetry into classical style.
Adapts to the APCD dataset.

Usage:
    python train_transformer.py --train_data ../data/processed --model_name AraGPT2 --epochs 5 --batch_size 4 --output_dir ../models/transformers
"""

import os
import argparse
import pandas as pd
from utils import create_transformer_model, train_model

def load_dataset_from_csv(processed_dir):
    """
    Loads the processed CSV files and converts them into a list of dictionaries.
    
    Args:
        processed_dir (str): Directory containing train.csv, valid.csv, test.csv
    
    Returns:
        list: List of training data entries.
    """
    train_path = os.path.join(processed_dir, "train.csv")
    valid_path = os.path.join(processed_dir, "valid.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df = pd.read_csv(train_path, header=None, names=['text', 'meter', 'era', 'poet', 'rhyme'])
    valid_df = pd.read_csv(valid_path, header=None, names=['text', 'meter', 'era', 'poet', 'rhyme'])
    test_df = pd.read_csv(test_path, header=None, names=['text', 'meter', 'era', 'poet', 'rhyme'])

    # Convert to list of dicts
    train_data = train_df.to_dict(orient='records')
    valid_data = valid_df.to_dict(orient='records')
    test_data = test_df.to_dict(orient='records')

    return train_data, valid_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer-based model.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed train.csv, valid.csv, test.csv.")
    parser.add_argument("--model_name", type=str, default="AraGPT2",
                        help="Name of the transformer model to use.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="../models/transformers",
                        help="Directory to save trained model checkpoints.")
    args = parser.parse_args()

    # Load data
    train_data, valid_data, test_data = load_dataset_from_csv(args.train_data)

    # Create or load transformer model
    model, tokenizer = create_transformer_model(args.model_name)

    # Train the model
    train_model(model, tokenizer, train_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir)

    print(f"Transformer training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
