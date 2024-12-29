# train_diffusion.py

"""
train_diffusion.py

Trains a diffusion model to iteratively refine text embeddings.
The model takes noisy text representations and refines them to produce classical Arabic poetry.

Usage:
    python train_diffusion.py --train_data ../data/processed --epochs 50 --batch_size 32 --output_dir ../models/diffusion
"""

import os
import argparse
import pandas as pd
from utils import create_diffusion_model, train_diffusion_model, load_encoder, get_input_encoded_data_h5

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
    parser = argparse.ArgumentParser(description="Train a diffusion model for Arabic poetry.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Directory containing processed train.csv, valid.csv, test.csv.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--output_dir", type=str, default="../models/diffusion",
                        help="Directory to save trained diffusion model checkpoints.")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="JSON string of model parameters.")
    args = parser.parse_args()

    # Load data
    train_data, valid_data, test_data = load_dataset_from_csv(args.train_data)

    # Define model parameters
    import json
    model_params = json.loads(args.model_params)

    # Define input shape based on preprocessed data
    sample_text = train_data[0]['text']
    max_bayt_len = 1000  # Adjust based on your preprocessing
    encoding_dim = 8  # Based on 8-bit encoding

    input_shape = (max_bayt_len, encoding_dim)

    # Create diffusion model
    diffusion_model = create_diffusion_model(input_shape, model_params)

    # Train the diffusion model
    history = train_diffusion_model(
        model=diffusion_model,
        train_data=train_data,
        valid_data=valid_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    print(f"Diffusion model training complete. Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
