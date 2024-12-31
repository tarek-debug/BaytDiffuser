# preprocess.py

"""
preprocess.py

This script handles data preprocessing tasks for the Arabic poetry project,
focusing on the APCD (Arabic Poetry Corpus Dataset).

Typical steps include:
- Reading raw APCD CSV data
- Filtering by era, meter, or other conditions
- Cleaning text (removing unwanted symbols, normalizing)
- Diacritization filtering
- Rhyme analysis (optional)
- Tokenizing / adding verse-level features
- Splitting data into train/validation/test
- Saving processed data

Usage:
    python preprocess.py --input_file ../data/raw/apcd_full.csv --output_dir ../data/processed
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your updated utils with new or modified functions
from utils import (
    check_percentage_tashkeel,
    Clean_data,
    separate_token_with_diacritics,
    factor_shadda_tanwin,
    extract_rhyme_info,          # New function
    get_verse_length_features    # New function
)

def filter_by_era_and_meter(df, allowed_eras=None, allowed_meters=None):
    """
    Optionally filters the DataFrame to only include poems from specified eras/meters.

    Args:
        df (pd.DataFrame): The input DataFrame.
        allowed_eras (list or None): List of eras to keep. If None, no era filtering is applied.
        allowed_meters (list or None): List of meters to keep. If None, no meter filtering is applied.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered = df.copy()

    if allowed_eras is not None and len(allowed_eras) > 0:
        filtered = filtered[filtered['العصر'].isin(allowed_eras)]

    if allowed_meters is not None and len(allowed_meters) > 0:
        filtered = filtered[filtered['البحر'].isin(allowed_meters)]

    return filtered

def add_additional_features(df):
    """
    Adds extra columns based on rhyme analysis, verse-length metrics, or other classical poem features.

    Args:
        df (pd.DataFrame): DataFrame containing Arabic poetry.

    Returns:
        pd.DataFrame: Updated DataFrame with new feature columns.
    """
    # Extract rhyme-based info from 'rhyme' or from 'combined_verse'
    df = extract_rhyme_info(df)

    # Add verse-level length features
    df = get_verse_length_features(df)

    return df

def preprocess_apcd_data(input_file, output_dir, tashkeel_threshold=0.4,
                         allowed_eras=None, allowed_meters=None):
    """
    Reads the APCD CSV file, cleans and tokenizes relevant text fields,
    applies diacritization filtering, splits the data, and saves as CSV.
    
    Additionally takes into account era, meter, rhyme, and verse length features.

    Args:
        input_file (str): Path to the raw APCD CSV file.
        output_dir (str): Directory to save processed data.
        tashkeel_threshold (float): Minimum percentage of diacritics required.
        allowed_eras (list or None): List of eras to retain (e.g., ['العصر الجاهلي', 'العصر الأموي', ...]).
        allowed_meters (list or None): List of meters to retain.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the APCD dataset (adjust encoding if necessary)
    df = pd.read_csv(input_file, encoding='utf-8')

    print(f"Original dataset size: {len(df)}")

    # Optionally filter by era/meter
    df = filter_by_era_and_meter(df, allowed_eras=allowed_eras, allowed_meters=allowed_meters)
    print(f"Dataset size after optional era/meter filtering: {len(df)}")

    # Create a new column that merges الشطر الايمن and الشطر الايسر as "combined_verse"
    df['combined_verse'] = df['الشطر الايمن'].astype(str) + " # " + df['الشطر الايسر'].astype(str)

    # Add extra feature columns (e.g., rhyme analysis, verse-length metrics)
    df = add_additional_features(df)

    # Apply diacritization filtering on 'البيت'
    print("Applying diacritization filtering...")
    df['passes_diacritization'] = df['البيت'].apply(
        lambda x: check_percentage_tashkeel(x, threshold=tashkeel_threshold)
    )
    filtered_df = df[df['passes_diacritization']].copy()
    print(f"Filtered dataset size (diacritics >= {tashkeel_threshold}): {len(filtered_df)}")

    # Select relevant columns
    # Add any newly generated columns that you want to keep (rhyme analysis, verse length, etc.)
    columns_to_keep = [
        'البيت', 'البحر', 'العصر', 'الشاعر', 'القافية',
        'combined_verse',
        'rhyme_info',         # New column from extract_rhyme_info
        'verse_length',       # New column from get_verse_length_features
        'avg_shatr_length'    # New column from get_verse_length_features
    ]
    # Only keep columns that exist to avoid KeyErrors
    columns_to_keep = [col for col in columns_to_keep if col in filtered_df.columns]

    processed_df = filtered_df[columns_to_keep].rename(columns={
        'البيت': 'text',
        'البحر': 'meter',
        'العصر': 'era',
        'الشاعر': 'poet',
        'القافية': 'rhyme'
    })
    print("Selected and renamed relevant columns.")

    # Clean the text
    print("Cleaning the text data...")
    processed_df = Clean_data(processed_df, max_bayt_len=1000, verse_column_name='text')
    print("Text data cleaned.")

    # Convert to DataFrame for the next stage
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
    parser.add_argument("--tashkeel_threshold", type=float, default=0.4,
                        help="Minimum percentage of diacritics required for a text to pass.")
    parser.add_argument("--allowed_eras", type=str, nargs='*', default=None,
                        help="List of eras to retain (e.g. العصر الجاهلي). If not specified, no era filtering is applied.")
    parser.add_argument("--allowed_meters", type=str, nargs='*', default=None,
                        help="List of meters to retain. If not specified, no meter filtering is applied.")

    args = parser.parse_args()

    preprocess_apcd_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        tashkeel_threshold=args.tashkeel_threshold,
        allowed_eras=args.allowed_eras,
        allowed_meters=args.allowed_meters
    )

if __name__ == "__main__":
    main()
