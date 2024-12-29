# evaluate.py

"""
evaluate.py

Evaluates model outputs against various metrics, such as:
- Meter / rhyme accuracy
- Lexical appropriateness
- Semantic preservation
- Comparison with baseline models

Usage:
    python evaluate.py --input_dir ../results/outputs --metrics_output ../data/metrics --diffusion_model_path ../models/diffusion/diffusion_model_final.h5 --transformer_model_path ../models/transformers/transformer_model_final.h5 --transformer_model_name 'aubmindlab/bert-base-arabertv2' --max_length 1000
"""

import os
import argparse
import json
from utils import (
    calculate_meter_score,
    calculate_rhyme_score,
    compare_with_baselines,
    load_diffusion_model,
    load_transformer_model,
    generate_classical_poem
)

def evaluate_model_outputs(input_dir, metrics_output, diffusion_model_path, transformer_model_path, transformer_model_name, max_length):
    """
    Evaluates each generated poem's adherence to classical constraints and
    optionally compares it with outputs from baseline models.
    
    Args:
        input_dir (str): Directory containing generated poems to evaluate.
        metrics_output (str): Directory to save evaluation results.
        diffusion_model_path (str): Path to the trained diffusion model.
        transformer_model_path (str): Path to the trained transformer model.
        transformer_model_name (str): Name of the transformer model used.
        max_length (int): Maximum sequence length used during training.
    
    Returns:
        None
    """
    # Ensure metrics output directory exists
    os.makedirs(metrics_output, exist_ok=True)

    # Load trained models
    transformer_model, tokenizer = load_transformer_model(transformer_model_path, transformer_model_name, max_length)
    diffusion_model = load_diffusion_model(diffusion_model_path)

    # Iterate through generated files
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                poem = f.read()

            # Calculate metrics
            meter_score = calculate_meter_score(poem)
            rhyme_score = calculate_rhyme_score(poem)

            results.append({
                "filename": filename,
                "meter_score": meter_score,
                "rhyme_score": rhyme_score
            })

    # Optional: Compare with baseline models
    compare_with_baselines(results)

    # Save results as JSON
    output_path = os.path.join(metrics_output, "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Results saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing generated poems to evaluate.")
    parser.add_argument("--metrics_output", type=str, required=True,
                        help="Directory to save evaluation results.")
    parser.add_argument("--diffusion_model_path", type=str, required=True,
                        help="Path to the trained diffusion model (.h5 file).")
    parser.add_argument("--transformer_model_path", type=str, required=True,
                        help="Path to the trained transformer model (.h5 file).")
    parser.add_argument("--transformer_model_name", type=str, required=True,
                        help="Name of the transformer model used (e.g., 'aubmindlab/bert-base-arabertv2').")
    parser.add_argument("--max_length", type=int, default=1000,
                        help="Maximum sequence length used during training.")
    args = parser.parse_args()

    evaluate_model_outputs(
        input_dir=args.input_dir,
        metrics_output=args.metrics_output,
        diffusion_model_path=args.diffusion_model_path,
        transformer_model_path=args.transformer_model_path,
        transformer_model_name=args.transformer_model_name,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()
