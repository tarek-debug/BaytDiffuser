"""
train_diffusion.py

Trains a diffusion model to iteratively refine text embeddings.
The model takes noisy text representations from the transformer output 
and refines them to produce classical Arabic poetry.

Usage:
    python train_diffusion.py --train_data ../data/processed --epochs 5 --batch_size 4 --output_dir ../models/diffusion
"""

import os
import argparse
import pandas as pd
from utils import create_diffusion_model, train_diffusion_model

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
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed train.csv, valid.csv, test.csv.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="../models/diffusion",
                        help="Directory to save trained diffusion model checkpoints.")
    args = parser.parse_args()

    # Load data
    train_data, valid_data, test_data = load_dataset_from_csv(args.train_data)

    # Create a diffusion model
    diffusion_model = create_diffusion_model()

    # Train the diffusion model
    train_diffusion_model(diffusion_model, train_data,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          output_dir=args.output_dir)

    print(f"Diffusion model training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
