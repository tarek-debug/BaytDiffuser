"""
preprocess.py

This script handles data preprocessing tasks for the Arabic poetry project,
focusing on the Arabic PCD (APCD) dataset.

Typical steps include:
- Reading raw APCD CSV data
- Extracting relevant fields (meter, poet, era, etc.)
- Cleaning text (removing unwanted symbols, normalizing)
- Tokenizing
- (Optional) Splitting data into train/validation/test
- Saving processed data

Usage:
    python preprocess.py --input_file ../data/raw/apcd/apcd_full.csv --output_dir ../data/processed
"""

import os
import argparse
import json
import pandas as pd

from utils import clean_text, tokenize_text

def preprocess_apcd_data(input_file, output_dir):
    """
    Reads the APCD CSV file, cleans and tokenizes relevant text fields,
    then saves the processed result as JSON in `output_dir`.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the APCD dataset (adjust encoding if necessary)
    df = pd.read_csv(input_file, encoding='utf-8')

    # Columns of interest (as discovered from your listing):
    #   العصر, الشاعر, الديوان, القافية, البحر, الشطر الايسر, الشطر الايمن, البيت
    # You can rename them if you want consistent column names in your code
    # For now, let's keep them in Arabic for clarity
    # Example: Combine الشطر الايمن and الشطر الايسر as "combined_verse"

    # Create a new column that merges the two half-lines (شطر أيمن / شطر أيسر) if desired:
    df['combined_verse'] = df['الشطر الايمن'].astype(str) + " " + df['الشطر الايسر'].astype(str)

    # Clean & tokenize the full "البيت" or "combined_verse" text
    # Here, we’ll choose the column `البيت` (the entire verse) for analysis.
    # Or you can use `combined_verse` if you prefer the merged shatr approach.
    cleaned_texts = []
    tokens_list = []

    for idx, row in df.iterrows():
        # Get the raw verse
        raw_verse = str(row['البيت'])  # adjust if you'd rather use 'combined_verse'
        
        # Clean the text
        cleaned = clean_text(raw_verse)
        
        # Tokenize
        tokens = tokenize_text(cleaned)

        cleaned_texts.append(cleaned)
        tokens_list.append(tokens)

    # Optionally store relevant metadata (meter, era, poet, etc.)
    # Let's store them as well in a final JSON structure
    processed_data = []
    for idx in range(len(df)):
        entry = {
            "original_verse": str(df['البيت'].iloc[idx]),
            "cleaned_verse": cleaned_texts[idx],
            "tokens": tokens_list[idx],
            "meter": str(df['البحر'].iloc[idx]),       # e.g., "الطويل"
            "era": str(df['العصر'].iloc[idx]),         # e.g., "قبل الإسلام"
            "poet": str(df['الشاعر'].iloc[idx]),       # e.g., "عمرو بنِ قُمَيئَة"
            "qafiya": str(df['القافية'].iloc[idx])     # e.g., "د"
        }
        processed_data.append(entry)

    # Save processed output as JSON
    output_filename = "apcd_processed.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(processed_data, out_f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessing complete. Processed file saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Arabic PCD (APCD) dataset.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the APCD CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save processed data.")
    args = parser.parse_args()

    preprocess_apcd_data(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
