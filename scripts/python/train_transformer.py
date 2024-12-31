# train_transformer.py

"""
train_transformer.py

This script trains a Transformer-based model (e.g., AraBERT, AraGPT2)
to transform modern Arabic poetry into classical style.
Adapts to the APCD dataset.

Usage:
    python train_transformer.py --train_data ../data/processed --model_name AraGPT2 --epochs 10 --batch_size 16 --output_dir ../models/transformers
"""

import os
import argparse
import pandas as pd
from BaytDiffuser.scripts.python.utils import create_transformer_model, train_transformer_model, load_encoder, get_input_encoded_data_h5

def load_dataset_from_csv(processed_dir):
    """
    Loads the processed CSV files and converts them into lists of dictionaries.
    
    Args:
        processed_dir (str): Directory containing train.csv, valid.csv, test.csv
    
    Returns:
        tuple: (train_data, valid_data, test_data)
    """
    train_path = os.path.join(processed_dir, "train.csv")
    valid_path = os.path.join(processed_dir, "valid.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    valid_df = pd.read_csv(valid_path, encoding='utf-8-sig')
    test_df = pd.read_csv(test_path, encoding='utf-8-sig')

    # Convert to list of dicts
    train_data = train_df.to_dict(orient='records')
    valid_data = valid_df.to_dict(orient='records')
    test_data = test_df.to_dict(orient='records')

    return train_data, valid_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer-based model for Arabic poetry.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed train.csv, valid.csv, test.csv.")
    parser.add_argument("--model_name", type=str, default="aubmindlab/bert-base-arabertv2",
                        help="Name of the pre-trained transformer model to use.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--output_dir", type=str, default="../models/transformers",
                        help="Directory to save trained transformer model checkpoints.")
    parser.add_argument("--max_length", type=int, default=1000,
                        help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    # Load data
    train_data, valid_data, test_data = load_dataset_from_csv(args.train_data)

    # Define max sequence length
    max_length = args.max_length

    # Define model parameters if needed
    # For simplicity, model parameters are defined within create_transformer_model

    # Create or load transformer model
    transformer_model, tokenizer = create_transformer_model(args.model_name, max_length)

    # Train the transformer model
    history = train_transformer_model(
        model=transformer_model,
        tokenizer=tokenizer,
        train_data=train_data,
        valid_data=valid_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_length=max_length
    )

    print(f"Transformer model training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
