"""
utils.py

Helper functions for preprocessing, model creation, training, evaluation, etc.
You can integrate Qawafi or other advanced libraries here for more
robust morphological or prosody analysis.

References:
 - Qawafi notebooks: they dissect poems thoroughly, handle diacritization, meter classification, etc.
   You can port their logic into these helper functions as needed.
"""

import re
import random
import json
import os

# ---------------------------------------------------------------------------
# Preprocessing Helpers
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Cleans and normalizes Arabic text by removing unwanted characters.
    You can expand this function with more advanced normalization as needed,
    e.g., using 'tnkeeh' or 'qawafi' preprocessing steps (remove tatweel, normalize Alef, etc.).
    """
    # Example: remove punctuation, non-Arabic letters, excessive whitespace
    text = re.sub(r"[^\u0621-\u063A\u0641-\u064A\s]+", "", text)  # keep Arabic chars and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(cleaned_text):
    """
    Simple whitespace-based tokenizer (placeholder).
    Replace with a more sophisticated tokenizer if desired
    (e.g. Qawafi morphological segmentation, or any HF tokenizer).
    """
    return cleaned_text.split()

def load_processed_data(directory_or_list):
    """
    Loads processed data from:
      - A directory containing one or more "_processed.json" files,
      - OR a list of file paths (strings).

    Returns a list of loaded JSON structures.
    """
    dataset = []

    # If it's a string and a directory, gather all matching files
    if isinstance(directory_or_list, str) and os.path.isdir(directory_or_list):
        json_files = sorted([
            f for f in os.listdir(directory_or_list)
            if f.endswith("_processed.json") or f.endswith(".json")
        ])
        for filename in json_files:
            file_path = os.path.join(directory_or_list, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data could be a list of entries or a single dict
                # We'll handle both:
                if isinstance(data, list):
                    dataset.extend(data)
                else:
                    dataset.append(data)
    elif isinstance(directory_or_list, list):
        # If it's a list of file paths
        for file_path in directory_or_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset.extend(data)
                else:
                    dataset.append(data)
    else:
        # Possibly a single file path string
        if os.path.isfile(directory_or_list):
            with open(directory_or_list, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset.extend(data)
                else:
                    dataset.append(data)

    return dataset

# ---------------------------------------------------------------------------
# Model Creation and Training Helpers
# ---------------------------------------------------------------------------
def create_transformer_model(model_name):
    """
    Create or load a pre-trained transformer model (e.g., from Hugging Face).
    Replace with real code to load your desired model/weights.
    """
    print(f"Loading transformer model: {model_name}")
    # Placeholder
    model = f"DummyTransformerModel({model_name})"
    tokenizer = f"DummyTokenizer({model_name})"
    return model, tokenizer

def train_model(model, tokenizer, dataset, epochs=3, batch_size=4, output_dir="checkpoints"):
    """
    Placeholder training loop for a transformer model.
    Replace with actual training code (e.g., PyTorch or Hugging Face Trainer).
    
    'dataset' is a list of dict entries with keys like:
        - 'original_verse'
        - 'cleaned_verse'
        - 'tokens'
        - 'meter'
        - 'era'
        - ...
    """
    print(f"Training model for {epochs} epochs with batch size {batch_size}...")
    for epoch in range(epochs):
        random.shuffle(dataset)
        # In a real setup, you'd:
        #  1. Convert tokens to input IDs
        #  2. Feed them into the model
        #  3. Calculate loss
        #  4. Update model weights
        print(f"Epoch {epoch+1}/{epochs}: ...")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/transformer_checkpoint.pt"
    print(f"Saving transformer model checkpoint to {checkpoint_path}")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        f.write("dummy checkpoint")

def create_diffusion_model():
    """
    Create or load a diffusion model.
    Replace with actual diffusion model code (e.g., text diffusion from Hugging Face).
    """
    diffusion_model = "DummyDiffusionModel"
    print("Loading diffusion model...")
    return diffusion_model

def train_diffusion_model(diffusion_model, dataset, epochs=3, batch_size=4, output_dir="checkpoints"):
    """
    Placeholder training loop for a diffusion model.
    Replace with actual training code.
    
    'dataset' is similar to above, a list of dict or tokens.
    """
    print(f"Training diffusion model for {epochs} epochs with batch size {batch_size}...")
    for epoch in range(epochs):
        random.shuffle(dataset)
        # Real logic: Add noise, denoise, etc.
        print(f"Epoch {epoch+1}/{epochs} for diffusion: ...")

    # Save diffusion model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/diffusion_checkpoint.pt"
    print(f"Saving diffusion model checkpoint to {checkpoint_path}")
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        f.write("dummy diffusion checkpoint")

def load_transformer_model():
    """
    Load trained transformer model from checkpoint.
    In practice, you'd return (model, tokenizer) after loading real files.
    """
    print("Loading trained transformer model from checkpoint...")
    return "TrainedTransformerModel", "TrainedTransformerTokenizer"

def load_diffusion_model():
    """
    Load trained diffusion model from checkpoint.
    """
    print("Loading trained diffusion model from checkpoint...")
    return "TrainedDiffusionModel"

# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def calculate_meter_score(poem):
    """
    Placeholder function that returns a random meter score between 0 and 1.
    For a real approach, you could adapt the Qawafi logic or a specialized
    علم العروض library that checks meter correctness.
    """
    return round(random.uniform(0, 1), 2)

def calculate_rhyme_score(poem):
    """
    Placeholder function returning a random rhyme accuracy score.
    For a real approach, see Qawafi's approach to extracting rhyme endings.
    """
    return round(random.uniform(0, 1), 2)

def compare_with_baselines(results):
    """
    Example function to compare your model's output with baseline outputs.
    Possibly load baseline model outputs from a folder and compare line-by-line.
    """
    print("Comparing with baseline models...")
    # Implement logic to compare results with baseline outputs
    pass

# ---------------------------------------------------------------------------
# Generation Helpers
# ---------------------------------------------------------------------------
def generate_classical_poem(modern_poem, transformer_model, transformer_tokenizer, diffusion_model):
    """
    Combines transformer and diffusion steps to produce a classical Arabic poem.
    - In real usage:
      1. Encode modern_poem with the transformer tokenizer.
      2. Generate an initial classical draft using the transformer model.
      3. Convert that draft to a latent/noisy representation.
      4. Refine it with the diffusion model.
      5. Decode final text form.

    Currently a placeholder.
    """
    return f"Classical version of: {modern_poem.strip()}"
