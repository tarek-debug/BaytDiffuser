# generate_output.py

"""
generate_output.py

Uses the trained Transformer and Diffusion models to generate Classical Arabic poems
from an input text (modern Arabic or any input prompt).

Usage:
    python generate_output.py --input_file ../data/examples/modern_poems.txt --output_dir ../results/outputs --transformer_model_path ../models/transformers/transformer_model_final.h5 --transformer_model_name 'aubmindlab/bert-base-arabertv2' --diffusion_model_path ../models/diffusion/diffusion_model_final.h5 --max_length 1000
"""

import os
import argparse
from utils import (
    load_transformer_model,
    load_diffusion_model,
    generate_classical_poem
)

def generate_output(input_file, output_dir, transformer_model_path, transformer_model_name, diffusion_model_path, max_length):
    """
    Generates classical Arabic poetry for each line in the input file
    using the Transformer and Diffusion models.
    
    Args:
        input_file (str): File containing modern Arabic poems (one per line).
        output_dir (str): Directory to save generated classical poems.
        transformer_model_path (str): Path to the trained transformer model (.h5 file).
        transformer_model_name (str): Name of the transformer model used (e.g., 'aubmindlab/bert-base-arabertv2').
        diffusion_model_path (str): Path to the trained diffusion model (.h5 file).
        max_length (int): Maximum sequence length used during training.
    
    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load trained models
    transformer_model, tokenizer = load_transformer_model(transformer_model_path, transformer_model_name, max_length)
    diffusion_model = load_diffusion_model(diffusion_model_path)

    # Read the modern poems (or any input lines)
    with open(input_file, 'r', encoding='utf-8') as f:
        modern_poems = f.readlines()

    # Generate classical versions
    for idx, poem in enumerate(modern_poems):
        poem = poem.strip()
        if not poem:
            continue  # Skip empty lines
        classical_version = generate_classical_poem(
            prompt=poem,
            transformer_model=transformer_model,
            tokenizer=tokenizer,
            diffusion_model=diffusion_model,
            max_length=max_length
        )

        # Save output
        output_path = os.path.join(output_dir, f"classical_poem_{idx+1}.txt")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(classical_version)

    print(f"Generated classical poems saved to {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Generate Classical Arabic poems from modern input.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="File containing modern Arabic poems (one per line).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated classical poems.")
    parser.add_argument("--transformer_model_path", type=str, required=True,
                        help="Path to the trained transformer model (.h5 file).")
    parser.add_argument("--transformer_model_name", type=str, required=True,
                        help="Name of the transformer model used (e.g., 'aubmindlab/bert-base-arabertv2').")
    parser.add_argument("--diffusion_model_path", type=str, required=True,
                        help="Path to the trained diffusion model (.h5 file).")
    parser.add_argument("--max_length", type=int, default=1000,
                        help="Maximum sequence length used during training.")
    args = parser.parse_args()

    generate_output(
        input_file=args.input_file,
        output_dir=args.output_dir,
        transformer_model_path=args.transformer_model_path,
        transformer_model_name=args.transformer_model_name,
        diffusion_model_path=args.diffusion_model_path,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()
