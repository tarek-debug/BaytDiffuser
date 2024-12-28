"""
generate_output.py

Uses the trained Transformer + Diffusion model to generate Classical Arabic poems
from modern Arabic inputs.

Usage:
    python generate_output.py --input_file ../data/examples/modern_poems.txt --output_dir ../results/outputs
"""

import os
import argparse
from utils import load_transformer_model, load_diffusion_model, generate_classical_poem

def generate_output(input_file, output_dir):
    """
    Generates classical Arabic poetry for each line in the input file
    using the hybrid Transformer + Diffusion pipeline.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load your trained models
    transformer_model, transformer_tokenizer = load_transformer_model()
    diffusion_model = load_diffusion_model()

    # Read the modern poems
    with open(input_file, 'r', encoding='utf-8') as f:
        modern_poems = f.readlines()

    # Generate classical versions
    for idx, poem in enumerate(modern_poems):
        classical_version = generate_classical_poem(
            poem, transformer_model, transformer_tokenizer, diffusion_model
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
    args = parser.parse_args()

    generate_output(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
