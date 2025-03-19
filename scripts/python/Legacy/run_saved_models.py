#!/usr/bin/env python
"""
run_saved_models.py

This script loads the saved models (e.g. your fine-tuned AraGPT2 model)
and runs a sample inference. Adjust the model directories as needed.
"""

import torch
from transformers import AutoTokenizer
# Import your custom model class if you need special behavior.
# (Make sure that the definition of AraGPT2ForClassicalStyle is in your PYTHONPATH.)
from scripts.python.your_model_definitions import AraGPT2ForClassicalStyle  # adjust import as needed

def load_transformer_model(model_dir: str):
    """
    Loads the AraGPT2 model and tokenizer from the given directory.
    Assumes that the model was saved via model.save_pretrained(model_dir)
    and tokenizer.save_pretrained(model_dir).
    """
    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print(f"Loading AraGPT2 model from {model_dir}")
    # Here we use the custom from_pretrained class method.
    model = AraGPT2ForClassicalStyle.from_pretrained(model_dir)
    return tokenizer, model

def run_inference(prompt: str, model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_transformer_model(model_dir)
    model.to(device)
    
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output using the underlying Hugging Face model.
    # You can adjust generation parameters as desired.
    with torch.no_grad():
        generated_ids = model.model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated output:")
    print(output_text)

if __name__ == "__main__":
    # Change this prompt as needed.
    sample_prompt = "يا ليل الصب متى غده"
    # Change the directory if your saved model is in another folder.
    transformer_model_dir = "./transformer_output"
    run_inference(sample_prompt, transformer_model_dir)
