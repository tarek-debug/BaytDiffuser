"""
evaluate.py

Evaluates model outputs against various metrics, such as:
- Meter / rhyme accuracy
- Lexical appropriateness
- Semantic preservation
- Comparison with baseline models

Usage:
    python evaluate.py --input_dir ../results/outputs --metrics_output ../data/metrics
"""

import os
import argparse
import json
from utils import calculate_meter_score, calculate_rhyme_score, compare_with_baselines

def evaluate_model_outputs(input_dir, metrics_output):
    """
    Evaluates each generated poem's adherence to classical constraints and
    optionally compares it with outputs from baseline models.
    """
    # Ensure metrics output directory exists
    os.makedirs(metrics_output, exist_ok=True)

    # Example: Iterate through generated files
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
    # E.g., compare_with_baselines(results)

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
    args = parser.parse_args()

    evaluate_model_outputs(args.input_dir, args.metrics_output)

if __name__ == "__main__":
    main()
