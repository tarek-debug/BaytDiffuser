# preprocess.py

"""
preprocess.py

This script handles data preprocessing tasks for the Arabic poetry project,
focusing on the APCD (Arabic Poetry Corpus Dataset).

Typical steps include:
- Reading raw APCD CSV data
- Cleaning text (removing unwanted symbols, normalizing)
- Diacritization filtering
- Tokenizing
- Splitting data into train/validation/test
- Saving processed data

Usage:
    python preprocess.py --input_file ../data/raw/apcd_full.csv --output_dir ../data/processed
"""

import os
import argparse
import pandas as pd
from utils import check_percentage_tashkeel, Clean_data, separate_token_with_diacritics, factor_shadda_tanwin

def preprocess_apcd_data(input_file, output_dir, tashkeel_threshold=0.7):
    """
    Reads the APCD CSV file, cleans and tokenizes relevant text fields,
    applies diacritization filtering, splits the data, and saves as CSV.
    
    Args:
        input_file (str): Path to the raw APCD CSV file.
        output_dir (str): Directory to save processed data.
        tashkeel_threshold (float): Minimum percentage of diacritics required.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the APCD dataset (adjust encoding if necessary)
    df = pd.read_csv(input_file, encoding='utf-8')

    # Columns of interest:
    # العصر, الشاعر, الديوان, القافية, البحر, الشطر الايسر, الشطر الايمن, البيت

    # Create a new column that merges الشطر الايمن and الشطر الايسر as "combined_verse"
    df['combined_verse'] = df['الشطر الايمن'].astype(str) + " # " + df['الشطر الايسر'].astype(str)

    # Apply diacritization filtering on 'البيت'
    print("Applying diacritization filtering...")
    df['passes_diacritization'] = df['البيت'].apply(lambda x: check_percentage_tashkeel(x, threshold=tashkeel_threshold))
    filtered_df = df[df['passes_diacritization']].copy()
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")

    # Select relevant columns
    processed_df = filtered_df[['البيت', 'البحر', 'العصر', 'الشاعر', 'القافية']]
    processed_df = processed_df.rename(columns={
        'البيت': 'text',
        'البحر': 'meter',
        'العصر': 'era',
        'الشاعر': 'poet',
        'القافية': 'rhyme'
    })

    # Clean the text
    print("Cleaning the text data...")
    processed_df = Clean_data(processed_df, max_bayt_len=1000, verse_column_name='text')  # Adjust max_bayt_len as needed

    # Convert to Hugging Face Dataset
    dataset = pd.DataFrame(processed_df)

    # Split the dataset
    print("Splitting the dataset into train, test, and validation...")
    train_df, temp_df = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save splits as CSV
    print("Saving the split datasets as CSV files...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False, encoding='utf-8-sig')
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')

    print(f"Preprocessing complete. Processed files saved to {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Arabic PCD (APCD) dataset.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the raw APCD CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save processed data.")
    parser.add_argument("--tashkeel_threshold", type=float, default=0.7,
                        help="Minimum percentage of diacritics required for a text to pass.")
    args = parser.parse_args()

    preprocess_apcd_data(args.input_file, args.output_dir, args.tashkeel_threshold)

if __name__ == "__main__":
    main()
