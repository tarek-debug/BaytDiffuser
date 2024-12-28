"""
preprocess.py

This script handles data preprocessing tasks for the Arabic poetry project.
Typical steps include:
- Reading raw data
- Cleaning text (removing unwanted symbols, normalizing)
- Tokenizing
- Splitting data into train/validation/test
- Saving processed data

Usage:
    python preprocess.py --input_dir ../data/raw --output_dir ../data/processed
"""

import os
import argparse
import json
from utils import clean_text, tokenize_text

def preprocess_data(input_dir, output_dir):
    """
    Reads raw data from `input_dir`, cleans and tokenizes it,
    then saves processed output to `output_dir`.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Example: Loop through files in input_dir
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)

            # 1. Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 2. Clean the text
            cleaned_text = clean_text(text)

            # 3. Tokenize the text
            tokens = tokenize_text(cleaned_text)

            # 4. Save the tokens in a processed file (e.g. JSON)
            output_filename = filename.replace(".txt", "_processed.json")
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(tokens, out_f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessing complete. Processed files saved to {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw Arabic poem data.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing raw data.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save processed data.")
    args = parser.parse_args()

    preprocess_data(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
