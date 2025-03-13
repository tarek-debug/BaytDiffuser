# utils.py

import os
import re
import sys
import errno
import gc
import pickle
import h5py
import math
import json
import numpy as np
import pandas as pd
from functools import partial, update_wrapper
from itertools import product
import unicodedata
from difflib import SequenceMatcher

# PyTorch and related libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# matplotlib

import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    pipeline,
    AutoModelForCausalLM
)

import subprocess

# Arabic-specific modules
import pyarabic.araby as araby


###############################################################################
#                          Global Constants & Utilities
###############################################################################

max_length = 128
max_bayt_len = 128  # for Diffusion, must match
encoding_dim = 8    # each token vector dimension

def filter_arabic(text):
    """
    Removes any non-Arabic characters (outside [\u0600-\u06FF]) and keeps spaces.
    """
    return re.sub(r'[^\u0600-\u06FF\s]', '', text)

###############################################################################
#                           Arabic References and Processing
###############################################################################

class Arabic:
    """
    Basic references for Arabic letters, diacritics, etc.
    """
    hamza            = u'\u0621'
    alef_mad         = u'\u0622'
    alef_hamza_above = u'\u0623'
    waw_hamza        = u'\u0624'
    alef_hamza_below = u'\u0625'
    yeh_hamza        = u'\u0626'
    alef             = u'\u0627'
    beh              = u'\u0628'
    teh_marbuta      = u'\u0629'
    teh              = u'\u062a'
    theh             = u'\u062b'
    jeem             = u'\u062c'
    hah              = u'\u062d'
    khah             = u'\u062e'
    dal              = u'\u062f'
    thal             = u'\u0630'
    reh              = u'\u0631'
    zain             = u'\u0632'
    seen             = u'\u0633'
    sheen            = u'\u0634'
    sad              = u'\u0635'
    dad              = u'\u0636'
    tah              = u'\u0637'
    zah              = u'\u0638'
    ain              = u'\u0639'
    ghain            = u'\u063a'
    feh              = u'\u0641'
    qaf              = u'\u0642'
    kaf              = u'\u0643'
    lam              = u'\u0644'
    meem             = u'\u0645'
    noon             = u'\u0646'
    heh              = u'\u0647'
    waw              = u'\u0648'
    alef_maksura     = u'\u0649'
    yeh              = u'\u064a'
    tatweel          = u'\u0640'

    # Diacritics
    fathatan         = u'\u064b'
    dammatan         = u'\u064c'
    kasratan         = u'\u064d'
    fatha            = u'\u064e'
    damma            = u'\u064f'
    kasra            = u'\u0650'
    shadda           = u'\u0651'
    sukun            = u'\u0652'

    # Lists
    alphabet = [
        alef, beh, teh, theh, jeem, hah, khah, dal, thal, reh, zain,
        seen, sheen, sad, dad, tah, zah, ain, ghain, feh, qaf, kaf,
        lam, meem, noon, heh, waw, yeh, hamza, alef_mad, alef_hamza_above,
        waw_hamza, alef_hamza_below, yeh_hamza, alef_maksura, teh_marbuta
    ]
    tashkeel = [fathatan, dammatan, kasratan, fatha, damma, kasra, sukun, shadda]


AR = Arabic()

###############################################################################
#                           Data Cleaning and Preprocessing
###############################################################################

def separate_token_with_diacritics(token):
    """
    Splits a string into a list of [char + diacritic].
    Example: "أَصبَحَ" -> ["أَ", "صْ", "بَ", "حَ"]
    """
    token = araby.strip_tatweel(token)
    hroof_with_tashkeel = []
    i = 0
    while i < len(token):
        if token[i] in AR.alphabet or token[i] == ' ' or token[i] == "\n":
            harf_with_taskeel = token[i]
            k = i
            while (k + 1 < len(token) and token[k+1] in AR.tashkeel):
                harf_with_taskeel += token[k+1]
                k += 1
            hroof_with_tashkeel.append(harf_with_taskeel)
            i = k + 1
        else:
            i += 1
    return hroof_with_tashkeel


def factor_shadda_tanwin(string):
    """
    Breaks shadda/tanwin combinations into multi-characters for standardization.
    """
    charsList = separate_token_with_diacritics(string)
    factoredString = ""
    for char in charsList:
        if len(char) < 2:
            factoredString += char
        elif len(char) == 2:
            base = char[0]
            dia = char[1]
            if dia in [AR.fatha, AR.damma, AR.kasra, AR.sukun]:
                factoredString += char
            elif dia == AR.dammatan:
                if base == AR.teh_marbuta:
                    factoredString += AR.teh + AR.damma + AR.noon + AR.sukun
                else:
                    factoredString += base + AR.damma + AR.noon + AR.sukun
            elif dia == AR.kasratan:
                if base == AR.teh_marbuta:
                    factoredString += AR.teh + AR.kasra + AR.noon + AR.sukun
                else:
                    factoredString += base + AR.kasra + AR.noon + AR.sukun
            elif dia == AR.fathatan:
                if base == AR.alef:
                    factoredString += AR.noon + AR.sukun
                elif base == AR.teh_marbuta:
                    factoredString += AR.teh + AR.fatha + AR.noon + AR.sukun
                else:
                    factoredString += base + AR.fatha + AR.noon + AR.sukun
            elif dia == AR.shadda:
                factoredString += base + AR.sukun + base
        else:
            # e.g. "شَّ" => base + AR.sukun + base + next_diacritic
            base = char[0]
            factoredString += base + AR.sukun + base + char[2]
    return factoredString


def Clean_data(processed_df, max_bayt_len, verse_column_name='text'):
    """
    Cleans and preprocesses a DataFrame containing Arabic poetry:
    - Normalize Hamza variations
    - Remove non-Arabic chars
    - Factor shadda/tanwin
    - Limit text length
    """
    processed_df['text'] = processed_df[verse_column_name].apply(lambda x: araby.normalize_hamza(x))
    processed_df['text'] = processed_df['text'].apply(lambda x: re.sub(r'[^\u0600-\u06FF\s]', '', x))
    processed_df['text'] = processed_df['text'].apply(factor_shadda_tanwin)
    processed_df = processed_df[processed_df['text'].apply(len) <= max_bayt_len]
    return processed_df

###############################################################################
#                         Meter Accuracy & Normalization
###############################################################################

def normalize_arabic(text):
    """
    Normalize Arabic text by standardizing letters, removing tatweel, etc.
    """
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)
    normalized_text = ''.join(
        char if unicodedata.category(char) != 'Mn' or char in ['ّ', 'ْ'] else ''
        for char in text
    )
    return normalized_text


def transliterate_arabic_to_syllables(verse):
    """
    Converts an Arabic verse into its syllabic structure (― for long, ∪ for short).
    """
    verse = normalize_arabic(verse)
    syllables = []
    vowels_short = ['َ', 'ِ', 'ُ']
    vowels_long = ['ا', 'و', 'ي']
    shadda = 'ّ'
    sukun = 'ْ'

    i = 0
    while i < len(verse):
        char = verse[i]
        next_char = verse[i+1] if i+1 < len(verse) else ''
        if re.match(r'[\u0621-\u064A]', char) and char not in vowels_long:
            if next_char == shadda:
                syllables.append('―')
                i += 2
                continue
            if next_char in vowels_short:
                syllables.append('∪')
                i += 2
                continue
            elif next_char in vowels_long:
                syllables.append('―')
                i += 2
                continue
            if next_char == sukun:
                syllables.append('―')
                i += 2
                continue
            syllables.append('∪')
            i += 1
        elif char in vowels_long:
            syllables.append('―')
            i += 1
        elif char in vowels_short:
            syllables.append('∪')
            i += 1
        else:
            i += 1

    return ''.join(syllables)


def calculate_meter_accuracy(verse, meter_pattern):
    """
    Compares the verse's syllabic structure to a target meter pattern.
    """
    verse_syllables = transliterate_arabic_to_syllables(verse)
    meter_syllables = meter_pattern.replace(' ', '')
    matcher = SequenceMatcher(None, verse_syllables, meter_syllables)
    accuracy = matcher.ratio() * 100
    return round(accuracy, 2)

###############################################################################
#                           Additional Feature Extraction
###############################################################################

def extract_rhyme_info(df):
    """
    Extract basic rhyme info from a 'rhyme' column or last words.
    """
    if 'rhyme' in df.columns:
        def get_rhyme_suffix(r):
            return r[-2:] if isinstance(r, str) and len(r) >= 2 else r
        df['rhyme_info'] = df['rhyme'].apply(get_rhyme_suffix)
    else:
        if 'combined_verse' in df.columns:
            def get_last_word_rhyme(text):
                lines = text.split('#')
                last_line = lines[-1].strip()
                last_word = last_line.split()[-1] if last_line.split() else ''
                return last_word[-2:] if len(last_word) >= 2 else last_word
            df['rhyme_info'] = df['combined_verse'].apply(get_last_word_rhyme)
        else:
            df['rhyme_info'] = None
    return df


def get_verse_length_features(df):
    """
    Adds columns about verse length, average shatr length, etc.
    """
    if 'combined_verse' in df.columns:
        df['verse_length'] = df['combined_verse'].apply(lambda v: len(v.replace('#', ' ')))
        def avg_shatr_len(v):
            parts = v.split('#')
            if len(parts) == 2:
                left_len = len(parts[0].strip())
                right_len = len(parts[1].strip())
                return (left_len + right_len) / 2
            return len(v)
        df['avg_shatr_length'] = df['combined_verse'].apply(avg_shatr_len)
    else:
        if 'الشطر الايسر' in df.columns and 'الشطر الايمن' in df.columns:
            def combined_len(row):
                left_len = len(str(row['الشطر الايسر']))
                right_len = len(str(row['الشطر الايمن']))
                return left_len + right_len
            df['verse_length'] = df.apply(combined_len, axis=1)
            df['avg_shatr_length'] = df['verse_length'] / 2
        else:
            df['verse_length'] = df['text'].apply(len)
            df['avg_shatr_length'] = df['text'].apply(len)
    return df

###############################################################################
#                      Tashkeel-Related Functions
###############################################################################

def get_alphabet_tashkeel_combination():
    base_letters = AR.alphabet + [' ']
    combos = [''] + base_letters[:]
    for letter in base_letters:
        for haraka in [AR.fatha, AR.damma, AR.kasra, AR.sukun]:
            combos.append(letter + haraka)
    return combos

lettersTashkeelCombination = get_alphabet_tashkeel_combination()
encoding_combination = [list(i) for i in product([0, 1], repeat=8)]

def string_with_tashkeel_vectorizer(string, padding_length):
    """
    Vectorizes a string with tashkeel into a (padding_length, 8) binary matrix.
    """
    factored_str = factor_shadda_tanwin(string)
    tokens = separate_token_with_diacritics(factored_str)
    representation = []
    for tok in tokens:
        if tok in lettersTashkeelCombination:
            idx = lettersTashkeelCombination.index(tok)
            representation.append(encoding_combination[idx])
        else:
            representation.append([0]*8)
    # pad to fixed length
    extra = padding_length - len(representation)
    for _ in range(extra):
        representation.append([0]*8)
    return np.array(representation, dtype=np.int32)

def oneHot_per_sample(string, padding_length):
    cleanedString = factor_shadda_tanwin(string)
    charCleanedString = separate_token_with_diacritics(cleanedString)
    encodedString = np.zeros((padding_length, len(lettersTashkeelCombination)), dtype=np.int32)
    letter = 0
    for char in charCleanedString:
        if char in lettersTashkeelCombination:
            one_index = lettersTashkeelCombination.index(char)
            encodedString[letter][one_index] = 1
        letter += 1
        if letter >= padding_length:
            break
    return encodedString

def oneHot_per_batch(batch_strings, padding_length):
    encodedBatch = np.zeros((len(batch_strings), padding_length, len(lettersTashkeelCombination)), dtype=np.int32)
    for i, string in enumerate(batch_strings):
        encoded = oneHot_per_sample(string, padding_length)
        encodedBatch[i] = encoded
    return encodedBatch

def check_percentage_tashkeel(string, threshold=0.4):
    implied_tashkeel_letters = {AR.alef, AR.waw, AR.yeh}
    total_chars = sum(1 for c in string if c.isalpha() and c not in implied_tashkeel_letters)
    if total_chars == 0:
        return False
    chars_with_diacritics = sum(
        1 for c in string if c not in implied_tashkeel_letters and c in AR.tashkeel
    )
    percentage = chars_with_diacritics / total_chars
    return percentage >= threshold

###############################################################################
#                           File I/O Utilities
###############################################################################

def save_h5(file_path, dataset_name, dataVar):
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=dataVar)
    print(f"Saved dataset '{dataset_name}' to '{file_path}'.")

def load_encoder(encoder_path):
    print(f"Loading encoder from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully.")
    return encoder

def encode_classes_data(categories):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(categories)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder

def decode_classes(onehot_vec, encoder):
    idx = np.argmax(onehot_vec)
    return encoder.inverse_transform([idx])[0]
    
###############################################################################
#                      Transformer (AraGPT2) Auto-Encoding for Classical Style
###############################################################################

class AraGPT2ForClassicalStyle(nn.Module):
    def __init__(self, model_name='aubmindlab/aragpt2-base', freeze_layers=0, dropout_prob=0.1):
        super(AraGPT2ForClassicalStyle, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Optionally freeze lower layers
        if freeze_layers > 0:
            for name, param in self.model.named_parameters():
                # Assuming naming convention like 'transformer.h.0'
                try:
                    layer_num = int(name.split('.')[2]) if len(name.split('.')) > 2 else -1
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                except (IndexError, ValueError):
                    continue
            print(f"Frozen the first {freeze_layers} layers of the AraGPT2 model.")
        
        # Optionally add a dropout layer before the final output
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.dropout(outputs.logits)
        return outputs.loss, logits
    
    def save_pretrained(self, save_directory):
        """
        Saves the underlying Hugging Face model and the custom dropout layer.
        """
        # Ensure the save directory exists and is a directory
        if os.path.exists(save_directory):
            if os.path.isfile(save_directory):
                os.remove(save_directory)
                print(f"Removed existing file at '{save_directory}' to create directory.")
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the Hugging Face model
        self.model.save_pretrained(save_directory)
        print(f"Underlying Hugging Face model saved to '{save_directory}'")
        
        # Save the custom dropout layer's state_dict
        dropout_path = os.path.join(save_directory, 'dropout.pt')
        torch.save(self.dropout.state_dict(), dropout_path)
        print(f"Custom dropout layer state_dict saved to '{dropout_path}'")
    
    @classmethod
    def from_pretrained(cls, save_directory, freeze_layers=0, dropout_prob=0.1):
        """
        Loads the underlying Hugging Face model and the custom dropout layer.
        """
        # Initialize the class without loading the model yet
        model_instance = cls(model_name=save_directory, freeze_layers=freeze_layers, dropout_prob=dropout_prob)
        
        # Load the Hugging Face model
        model_instance.model = AutoModelForCausalLM.from_pretrained(save_directory)
        print(f"Underlying Hugging Face model loaded from '{save_directory}'")
        
        # Load the custom dropout layer's state_dict
        dropout_path = os.path.join(save_directory, 'dropout.pt')
        if os.path.exists(dropout_path):
            model_instance.dropout.load_state_dict(torch.load(dropout_path, map_location='cpu'))
            print(f"Custom dropout layer state_dict loaded from '{dropout_path}'")
        else:
            print(f"No custom dropout layer found at '{dropout_path}'. Using default dropout.")
        
        return model_instance

def train_aragpt2_for_classical_style(df_classical, tokenizer, model,
                                      max_length=128, epochs=10, batch_size=4,
                                      output_dir='./transformer_output', device='cuda' if torch.cuda.is_available() else 'cpu',
                                      freeze_layers=0, weight_decay=0.01, 
                                      patience=3, max_grad_norm=1.0):
    """
    Trains the AraGPT2 model for refining classical poems with improved generalization.

    Args:
        df_classical (pd.DataFrame): DataFrame containing classical poems with a 'text' column.
        tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        model (AraGPT2ForClassicalStyle): Compiled AraGPT2 model.
        max_length (int): Maximum sequence length.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        output_dir (str): Directory to save training logs and checkpoints.
        device (str): Device to train on ('cuda' or 'cpu').
        freeze_layers (int): Number of lower layers to freeze.
        weight_decay (float): Weight decay (L2 regularization) coefficient.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        model: Trained AraGPT2 model.
        history: Training history dictionary containing loss and perplexity metrics.
    """
    import os
    from sklearn.model_selection import train_test_split
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Prepare the data
    texts = df_classical['text'].tolist()
    encodings = tokenizer(
        texts, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:].clone()
    labels[:, -1] = tokenizer.eos_token_id

    # Split the data into training and validation sets
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Move model to device
    model.to(device)

    # Define optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=weight_decay)

    # Define learning rate scheduler without the deprecated 'verbose' parameter
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Prepare for training
    history = {'train_loss': [], 'val_loss': [], 'train_perplexity': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(output_dir, exist_ok=True)

    # Initialize current_lr for tracking changes
    current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * input_ids_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_perplexity = math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf')
        history['train_loss'].append(avg_train_loss)
        history['train_perplexity'].append(train_perplexity)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
                loss, logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                val_loss += loss.item() * input_ids_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        history['val_loss'].append(avg_val_loss)
        history['val_perplexity'].append(val_perplexity)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_perplexity:.2f} | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Check for learning rate change
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning rate adjusted from {current_lr} to {new_lr}")
            current_lr = new_lr

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = os.path.join(output_dir, f"model_epoch_{epoch+1}")
            # Remove existing directory if it exists as a file
            if os.path.exists(checkpoint_dir) and os.path.isfile(checkpoint_dir):
                os.remove(checkpoint_dir)
                print(f"Removed existing file at '{checkpoint_dir}' to create directory.")
            model.save_pretrained(checkpoint_dir)
            print(f"Saved best model to '{checkpoint_dir}'")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    return model, history


def inference_convert_classical(classical_verse, tokenizer, model, max_length=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Converts a diacritized classical verse using the AraGPT2 model.

    Args:
        classical_verse (str): Diacritized classical Arabic verse.
        tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        model (AraGPT2ForClassicalStyle): Trained AraGPT2 model.
        max_length (int): Maximum sequence length.
        device (str): Device to perform inference on.

    Returns:
        str: Refined classical verse.
    """
    model.eval()
    inputs = tokenizer(
        classical_verse,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        # Use max_new_tokens instead of max_length to prevent exceeding input size
        generated_ids = model.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # Adjust as needed
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=1.0
        )
    
    refined_verse = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return refined_verse

###############################################################################
#            NEW: ThePoet Integration (mabaji/thepoet) for Initial Gen
###############################################################################

def add_tashkeel_with_java(verse):
    """
    Calls the VerseDiacritizer Java program to diacritize an Arabic verse.
    Args:
        verse (str): Arabic text to be diacritized.
    Returns:
        str: Diacritized Arabic text.
    """
    original_cwd = os.getcwd()  # Save the current working directory
    try:
        # Change to the directory where Java dependencies are located
        os.chdir("../scripts/java")
        
        # Command to call the Java program
        command = [
            "java",
            "-cp",
            ".:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/FarasaDiacritizeJar.jar:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/lib/*",
            "VerseDiacritizer",
            verse
        ]
        
        # Execute the Java program and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            return result.stdout.strip()  # Diacritized text
        else:
            print("Error in diacritization:", result.stderr)
            return verse  # Return the original verse in case of an error

    except Exception as e:
        print(f"Exception occurred: {e}")
        return verse  # Return the original verse in case of an exception

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)



def create_thepoet_pipeline():
    """
    Creates a text-generation pipeline using the mabaji/thepoet model 
    for generating Arabic poetry.
    """
    poet_tokenizer = AutoTokenizer.from_pretrained("mabaji/thepoet")
    poet_model = AutoModelForCausalLM.from_pretrained("mabaji/thepoet")
    poet_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    poet_pipeline = pipeline(
        "text-generation", 
        model=poet_model, 
        tokenizer=poet_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return poet_pipeline


def generate_rough_poem_with_thepoet(prompt, poet_pipeline, max_length=50, num_return_sequences=1, max_attempts=10):
    """
    Generates a rough classical Arabic poem based on the given prompt
    using the mabaji/thepoet pipeline. Ensures the following format:
        - First verse matches the input.
        - Each subsequent verse has two halves separated by '   '.
        - Verses are separated by '.'.
    Enhancements:
        - Accumulates verses until at least six are achieved.
        - If format issues arise, takes the last verse and feeds it again to generate new verses.
        - If after six verses issues persist, trims the last verse before the period.
        - Adds missing periods to complete verses.
    """
    import re

    verses = []
    attempts = 0
    max_attempts = max_attempts  # Total attempts allowed
    desired_verse_count = 6

    current_prompt = prompt

    while len(verses) < desired_verse_count and attempts < max_attempts:
        try:
            # Generate poem using ThePoet
            results = poet_pipeline(
                current_prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,      # Enable sampling for creativity
                temperature=0.8      # Adjust temperature as desired
            )
            generated_texts = [r["generated_text"] for r in results]
            print("\nGenerated rough poems from ThePoet:")
            for idx, text in enumerate(generated_texts, 1):
                print(f"{idx}: {text}")

            # Process each generated text
            for text in generated_texts:
                # Split the generated text into verses based on '.'
                split_verses = text.split('.')
                split_verses = [v.strip() for v in split_verses if v.strip()]

                # Validate and append verses
                for verse in split_verses:
                    # Ensure the first verse matches the prompt
                    if not verses and verse != prompt:
                        print("First verse does not match the prompt. Skipping.")
                        continue

                    # Skip the prompt if already added
                    if not verses and verse == prompt:
                        verses.append(verse)
                        continue

                    # Check if verse has two halves separated by '   ' or '-'
                    if '   ' not in verse and '-' not in verse:
                        # Attempt to split the verse into two halves
                        words = verse.split()
                        if len(words) < 2:
                            print(f"Verse too short to split into halves: '{verse}'. Skipping.")
                            continue
                        mid_point = len(words) // 2
                        first_half = ' '.join(words[:mid_point])
                        second_half = ' '.join(words[mid_point:])
                        fixed_verse = f"{first_half}   {second_half}"

                        print(f"Verse does not have two halves. Fixed verse: {fixed_verse}")

                        # Ensure the fixed verse ends with a period
                        if not fixed_verse.endswith('.'):
                            fixed_verse += '.'

                        verses.append(fixed_verse)
                    else:
                        # Replace hyphen with three spaces if present
                        if '-' in verse:
                            fixed_verse = verse.replace('-', '   ')
                            print(f"Replaced hyphen with three spaces: {fixed_verse}")
                        else:
                            fixed_verse = verse

                        # Ensure the verse ends with a period
                        if not fixed_verse.endswith('.'):
                            fixed_verse += '.'
                            print(f"Added missing period to verse: {fixed_verse}")

                        verses.append(fixed_verse)

                    # Update the current prompt to the last verse for next generation
                    current_prompt = verse

                    # Check if we've reached the desired number of verses
                    if len(verses) >= desired_verse_count:
                        break

            # Increment attempts if desired verses not yet achieved
            if len(verses) < desired_verse_count:
                attempts += 1
                print(f"Attempt {attempts}: Generated poems do not match the required format. Regenerating...")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            attempts += 1
            continue

    # Post-processing to ensure at least six verses
    if len(verses) < desired_verse_count and verses:
        print("\nFinalizing the poem by trimming the last incomplete verse if necessary.")
        last_verse = verses[-1]
        if '.' in last_verse:
            # Trim the last verse before the period
            trimmed_verse = last_verse.rsplit('.', 1)[0].strip()
            if '   ' in trimmed_verse:
                verses[-1] = trimmed_verse + '.'
                print(f"Trimmed last verse: {verses[-1]}")
            else:
                # If still not in the correct format, split into halves
                words = trimmed_verse.split()
                if len(words) < 2:
                    verses[-1] = trimmed_verse + '.'
                    print(f"Trimmed last verse (cannot split further): {verses[-1]}")
                else:
                    mid_point = len(words) // 2
                    fixed_verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:]) + '.'
                    verses[-1] = fixed_verse
                    print(f"Fixed last verse by splitting into halves: {verses[-1]}")

    # Ensure all verses end with a period
    for i, verse in enumerate(verses):
        if not verse.endswith('.'):
            verses[i] = verse + '.'
            print(f"Added missing period to verse {i+1}: {verses[i]}")

    # Ensure exactly six verses by trimming or padding if necessary
    if len(verses) > desired_verse_count:
        verses = verses[:desired_verse_count]
        print(f"\nTrimmed the poem to the first {desired_verse_count} verses.")

    elif len(verses) < desired_verse_count:
        # Pad with empty verses if still less than six
        while len(verses) < desired_verse_count:
            verses.append(".")
            print(f"Added empty verse to reach six verses.")

    # Combine the verses into the final poem
    final_poem = '\n'.join(verses)
    print(f"\n==== Final Chained Poem ====\n{final_poem}\n================================")

    return final_poem

def generate_classical_poem_with_thepoet(
    modern_prompt,
    poet_pipeline,
    transformer_model,
    transformer_tokenizer,
    # diffusion_model,  # Removed as we're skipping Diffusion
    max_length=128,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generates a refined classical Arabic poem by chaining ThePoet and AraGPT2.

    Steps:
    1. Generate a rough poem using ThePoet.
    2. Split the poem into verses.
    3. Diacritize each verse.
    4. Refine each diacritized verse using AraGPT2.
    5. (Optional) Further refine with Diffusion model.
    6. Combine all refined verses into the final poem.

    Args:
        modern_prompt (str): Modern Arabic prompt to initiate poem generation.
        poet_pipeline (transformers.pipeline): ThePoet pipeline for initial generation.
        transformer_model (AraGPT2ForClassicalStyle): Fine-tuned AraGPT2 model.
        transformer_tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        diffusion_model (tf.keras.Model, optional): Diffusion model for further refinement.
        max_length (int): Maximum sequence length for models.
        device (str): Device to perform inference on.

    Returns:
        str: Final refined classical Arabic poem.
    """
    import re

    # Step 1: Generate rough poem with ThePoet
    rough_poem = generate_rough_poem_with_thepoet(
        prompt=modern_prompt,
        poet_pipeline=poet_pipeline,
        max_length=50,
        num_return_sequences=1
    )
    print(f"\nRough Poem from ThePoet:\n{rough_poem}")

    # Step 2: Replace hyphens with three spaces and normalize spaces
    rough_poem = re.sub(r'[-–—]', '   ', rough_poem)
    rough_poem = re.sub(r'\s+', ' ', rough_poem).strip()
    print(f"\nRough Poem after formatting:\n{rough_poem}")

    # Step 3: Split into verses based on punctuation
    verse_delimiters = r'[.!؟]'
    verses = re.split(verse_delimiters, rough_poem)
    verses = [verse.strip() for verse in verses if verse.strip()]
    print(f"\nNumber of verses extracted: {len(verses)}")
    for idx, verse in enumerate(verses, 1):
        print(f"Verse {idx}: {verse}")

    # Step 4: Process each verse
    processed_verses = []
    for idx, verse in enumerate(verses, 1):
        print(f"\nProcessing Verse {idx}: {verse}")

        # Ensure two halves separated by three spaces
        if '   ' not in verse:
            words = verse.split()
            if len(words) < 2:
                print(f"Verse too short to split into halves: '{verse}'. Skipping.")
                continue
            mid_point = len(words) // 2
            verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
            print(f"Verse split into halves: {verse}")

        # Step 4a: Diacritize
        diacritized_verse = add_tashkeel_with_java(verse)
        print(f"Diacritized Verse: {diacritized_verse}")

        # Step 4b: Ensure formatting post-diacritization
        if '   ' not in diacritized_verse:
            words = diacritized_verse.split()
            if len(words) < 2:
                diacritized_verse = diacritized_verse + '.'
                print(f"Diacritized Verse too short to split further. Added period: {diacritized_verse}")
            else:
                mid_point = len(words) // 2
                diacritized_verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
                print(f"Diacritized Verse re-split into halves: {diacritized_verse}")
        else:
            diacritized_verse = re.sub(r'\s+', '   ', diacritized_verse).strip()
            print(f"Diacritized Verse normalized spaces: {diacritized_verse}")

        # Step 4c: Transformer (AraGPT2) Refinement
        classical_draft = inference_convert_classical(
            classical_verse=diacritized_verse,
            tokenizer=transformer_tokenizer,
            model=transformer_model,
            max_length=max_length,
            device=device
        )
        print(f"Classical Draft from AraGPT2: {classical_draft}")

        # **Removed Step 4d: Diffusion Refinement**
        # Since we're skipping the Diffusion model, we directly use the Transformer output.

        # Optionally, you can still pass the classical_draft through `filter_arabic` if needed
        final_verse = filter_arabic(classical_draft)
        print(f"Final Verse after AraGPT2 Refinement: {final_verse}")

        # Append to processed verses
        if final_verse:
            processed_verses.append(final_verse)
        else:
            print(f"Warning: Final verse {idx} is empty after AraGPT2 refinement.")

    # Step 5: Combine into final poem
    final_poem = '\n'.join(processed_verses)
    print(f"\n==== Final Chained Poem ====\n{final_poem}\n================================")
    return final_poem

###############################################################################
#                     Sequence-based Batch Generator
###############################################################################

class ShaarDataset(Dataset):
    def __init__(self, bayt_texts, bhore_texts, vectorize_fun, max_bayt_len):
        self.bayt_texts = bayt_texts.reset_index(drop=True)
        self.bhore_texts = bhore_texts
        self.vectorize_fun = vectorize_fun
        self.max_bayt_len = max_bayt_len

    def __len__(self):
        return len(self.bayt_texts)

    def __getitem__(self, idx):
        batch_text = self.bayt_texts[idx]
        batch_y = self.bhore_texts[idx]
        X_enc = self.vectorize_fun(batch_text, self.max_bayt_len)
        return torch.tensor(X_enc, dtype=torch.float32), torch.tensor(batch_y, dtype=torch.float32)

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

###############################################################################
#                     Logging and Checkpoint Utilities
###############################################################################

def update_log_file(exp_name, text, epoch_flag=False):
    def _update_line(line, newtext, ep_flag):
        if ep_flag:
            prefix, epoch_count = line.split("@")
            epoch_count = str(int(epoch_count) + 1)
            return prefix + "@" + epoch_count
        else:
            prefix = line.split(",")[0]
            return prefix + "," + newtext

    try:
        if not os.path.exists("log.txt"):
            with open("log.txt", 'w') as f:
                f.write(f"{exp_name}, {text}\n")
            return True

        lines = open("log.txt").read().split('\n')
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            if exp_name == line.split(",")[0]:
                new_line = _update_line(line, text, epoch_flag)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        with open("log.txt", 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    except Exception as e:
        print(f"Error updating log file: {e}")
        return False

def remove_non_max(checkpoints_path):
    try:
        models = os.listdir(checkpoints_path)
        metric_files = []
        for filename in models:
            if 'weights-improvement' in filename and filename.endswith('.h5'):
                parts = filename.split('-')
                try:
                    metric = float(parts[-2])
                    metric_files.append((filename, metric))
                except:
                    continue
        if not metric_files:
            print("No checkpoint files found to evaluate.")
            return
        metric_files_sorted = sorted(metric_files, key=lambda x: x[1])
        best_file = metric_files_sorted[0][0]
        print(f"Best checkpoint identified: {best_file}")
        for file, _ in metric_files_sorted[1:]:
            os.remove(os.path.join(checkpoints_path, file))
            print(f"Removed checkpoint: {file}")
    except Exception as e:
        print(f"Error in removing non-max checkpoints: {e}")


###############################################################################
#                      Evaluation Utilities
###############################################################################

def recall_precision_f1(confusion_matrix_df):
    confusion_matrix_np = confusion_matrix_df.values
    num_classes = confusion_matrix_np.shape[0]
    recall_per_class = np.diag(confusion_matrix_np) / np.sum(confusion_matrix_np, axis=1)
    precision_per_class = np.diag(confusion_matrix_np) / np.sum(confusion_matrix_np, axis=0)
    recall_per_class = np.nan_to_num(recall_per_class)
    precision_per_class = np.nan_to_num(precision_per_class)
    f1_scores = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    f1_scores = np.nan_to_num(f1_scores)
    f1_score = np.mean(f1_scores)
    results_df = pd.DataFrame({
        'Recall': recall_per_class,
        'Precision': precision_per_class
    }, index=confusion_matrix_df.index)
    return results_df, f1_score


###############################################################################
#                          Diffusion Model in PyTorch
###############################################################################

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, model_params):
        """
        Initializes the Diffusion Model with Transformer blocks.

        Args:
            input_dim (int): Dimension of the input features.
            model_params (dict): Dictionary containing model parameters:
                - num_transformer_blocks (int)
                - num_heads (int)
                - key_dim (int)
                - ffn_units (int)
        """
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_transformer_blocks = model_params.get('num_transformer_blocks', 2)
        self.num_heads = model_params.get('num_heads', 4)
        self.key_dim = model_params.get('key_dim', 64)
        self.ffn_units = model_params.get('ffn_units', 256)
        
        # Initial LayerNorm
        self.initial_norm = nn.LayerNorm(input_dim)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                input_dim=input_dim,
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                ffn_units=self.ffn_units
            )
            for _ in range(self.num_transformer_blocks)
        ])
        
        # Output Layer
        self.output_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        """
        Forward pass through the Diffusion Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim)
        """
        x = self.initial_norm(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, key_dim, ffn_units):
        """
        Initializes a single Transformer block.

        Args:
            input_dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            key_dim (int): Dimension of the keys in attention.
            ffn_units (int): Number of units in the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_units),
            nn.ReLU(),
            nn.Linear(ffn_units, input_dim)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Forward pass through a Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Self-Attention with Residual Connection
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-Forward Network with Residual Connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x

def create_diffusion_model_pytorch(input_shape, model_params):
    """
    Creates the Diffusion Model in PyTorch.

    Args:
        input_shape (tuple): Shape of the input (seq_len, encoding_dim)
        model_params (dict): Parameters for the model architecture.

    Returns:
        DiffusionModel: An instance of the DiffusionModel class.
    """
    seq_len, encoding_dim = input_shape
    model = DiffusionModel(input_dim=encoding_dim, model_params=model_params)
    return model

###############################################################################
#                    Training Function for Diffusion Model
###############################################################################

class DiffusionDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initializes the Dataset for Diffusion Model.

        Args:
            X (torch.Tensor): Input tensor of shape (num_samples, seq_len, encoding_dim)
            Y (torch.Tensor): Target tensor of shape (num_samples, seq_len, encoding_dim)
        """
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def embed_tokens_pytorch(input_ids, max_bayt_len, encoding_dim):
    """
    Embeds input IDs by repeating them along the last dimension.

    Args:
        input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)
        max_bayt_len (int): Maximum sequence length.
        encoding_dim (int): Dimension to repeat each token.

    Returns:
        torch.Tensor: Embedded tensor of shape (batch_size, max_bayt_len, encoding_dim)
    """
    batch_size, seq_len = input_ids.size()
    if seq_len < max_bayt_len:
        padding = torch.zeros((batch_size, max_bayt_len - seq_len), dtype=input_ids.dtype, device=input_ids.device)
        input_ids_padded = torch.cat([input_ids, padding], dim=1)
    else:
        input_ids_padded = input_ids[:, :max_bayt_len]
    
    # Repeat the input IDs along the last dimension
    embedded = input_ids_padded.unsqueeze(-1).repeat(1, 1, encoding_dim)
    return embedded

def train_diffusion_for_classical_style_pytorch(df_classical, diffusion_model, tokenizer,
                                               max_length=128, max_bayt_len=128, encoding_dim=8,
                                               epochs=5, batch_size=4, output_dir='./diffusion_output',
                                               learning_rate=1e-3, patience=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains the Diffusion Model in PyTorch for classical style refinement.

    Args:
        df_classical (pd.DataFrame): DataFrame containing classical poems with a 'text' column.
        diffusion_model (DiffusionModel): Initialized Diffusion Model.
        tokenizer (AutoTokenizer): Tokenizer to process the text.
        max_length (int): Maximum sequence length for tokenization.
        max_bayt_len (int): Maximum sequence length after embedding.
        encoding_dim (int): Dimension to embed each token.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        output_dir (str): Directory to save model checkpoints.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement for early stopping.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        DiffusionModel: Trained Diffusion Model.
        dict: Training history containing loss metrics.
    """
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the data
    texts = df_classical['text'].tolist()
    encodings = tokenizer(
        texts, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids']  # Shape: (num_samples, seq_len)
    labels = input_ids.clone()
    
    # Embed tokens
    embedded_X = embed_tokens_pytorch(input_ids, max_bayt_len, encoding_dim)  # Shape: (num_samples, max_bayt_len, encoding_dim)
    embedded_Y = embedded_X.clone()
    
    # Create Dataset and DataLoader
    dataset = DiffusionDataset(embedded_X, embedded_Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to device
    diffusion_model.to(device)
    
    # Define optimizer and loss function
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Training Loop with Early Stopping and Checkpointing
    best_val_mae = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    
    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        train_loss = 0.0
        train_mae = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)  # Shape: (batch_size, max_bayt_len, encoding_dim)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            outputs = diffusion_model(batch_X)  # Shape: (batch_size, max_bayt_len, encoding_dim)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_mae += F.l1_loss(outputs, batch_Y, reduction='sum').item()
        
        avg_train_loss = train_loss / train_size
        avg_train_mae = train_mae / train_size
        
        # Validation
        diffusion_model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                outputs = diffusion_model(batch_X)
                loss = criterion(outputs, batch_Y)
                
                val_loss += loss.item() * batch_X.size(0)
                val_mae += F.l1_loss(outputs, batch_Y, reduction='sum').item()
        
        avg_val_loss = val_loss / val_size
        avg_val_mae = val_mae / val_size
        
        # Logging
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('MAE/Train', avg_train_mae, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('MAE/Validation', avg_val_mae, epoch)
        
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # Check for improvement
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            checkpoint_path = os.path.join(output_dir, f"diffusion_model_epoch_{epoch}.pt")
            torch.save(diffusion_model.state_dict(), checkpoint_path)
            print(f"Saved best model to '{checkpoint_path}'")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation MAE for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training complete.")
    return diffusion_model, history


def preprocess_texts(texts, preprocessor):
    """
    Applies ArabertPreprocessor to a list of texts.

    Args:
        texts (list of str): Raw text data.
        preprocessor (ArabertPreprocessor): Preprocessor instance.

    Returns:
        list of str: Preprocessed text data.
    """
    return [preprocessor.preprocess(text) for text in texts]
###############################################################################
#                        Denoising Auto-Encoder (DAE)
###############################################################################

class DenoisingAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, latent_dim=256, max_length=128):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        
        self.decoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embedding_dim, latent_dim, batch_first=True)
        self.fc = nn.Linear(latent_dim, vocab_size)
    
    def forward(self, encoder_input, decoder_input):
        embedded = self.encoder(encoder_input)
        _, (hidden, cell) = self.encoder_lstm(embedded)
        
        embedded_dec = self.decoder(decoder_input)
        outputs, _ = self.decoder_lstm(embedded_dec, (hidden, cell))
        logits = self.fc(outputs)
        return logits

def create_denoising_autoencoder(vocab_size, embedding_dim=128, latent_dim=256, max_length=128):
    model = DenoisingAutoencoder(vocab_size, embedding_dim, latent_dim, max_length)
    return model

def create_noise(input_text, drop_prob=0.2):
    output = []
    for ch in input_text:
        if np.random.rand() < drop_prob:
            continue
        else:
            output.append(ch)
    return "".join(output)

def train_denoising_autoencoder(texts, tokenizer, max_length=128, test_size=0.2, 
                                embedding_dim=128, latent_dim=256,
                                epochs=5, batch_size=4, output_dir="./dae_output", device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    X_noisy = [create_noise(t) for t in texts]
    Y_clean = texts

    enc_noisy = tokenizer(X_noisy, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    enc_clean = tokenizer(Y_clean, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    X_ids = enc_noisy["input_ids"]
    Y_ids = enc_clean["input_ids"]

    dataset = torch.utils.data.TensorDataset(X_ids, Y_ids)
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size
    dae_model = create_denoising_autoencoder(vocab_size, embedding_dim, latent_dim, max_length).to(device)

    optimizer = Adam(dae_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_loss = float('inf')
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        dae_model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            noisy, clean = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = dae_model(noisy, noisy)
            logits = outputs.view(-1, vocab_size)
            clean = clean.view(-1)
            loss = criterion(logits, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            mask = clean != tokenizer.pad_token_id
            correct += (preds == clean).masked_select(mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = (correct / total) * 100
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        # Validation
        dae_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = [b.to(device) for b in batch]
                outputs = dae_model(noisy, noisy)
                logits = outputs.view(-1, vocab_size)
                clean = clean.view(-1)
                loss = criterion(logits, clean)
                val_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                mask = clean != tokenizer.pad_token_id
                val_correct += (preds == clean).masked_select(mask).sum().item()
                val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f"dae_model_epoch_{epoch+1}.pt")
            torch.save(dae_model.state_dict(), checkpoint_path)
            print(f"Saved best DAE model to '{checkpoint_path}'")

    final_model_path = os.path.join(output_dir, "denoising_autoencoder_final.pt")
    torch.save(dae_model.state_dict(), final_model_path)
    print(f"DAE model saved to '{final_model_path}'")
    return dae_model, history

###############################################################################
#                        Masked Language Modeling (MLM)
###############################################################################

def train_masked_language_model(texts, model_name="aubmindlab/bert-base-arabertv2",
                                max_length=128, epochs=3, batch_size=2, output_dir="./mlm_output", device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    encodings = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    train_ids, val_ids, train_mask, val_mask = train_test_split(
        input_ids, attention_mask, test_size=0.2, random_state=42
    )

    class MLMDataset(Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx]
            }

    train_dataset = MLMDataset(train_ids, train_mask)
    val_dataset = MLMDataset(val_ids, val_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    optimizer = Adam(model.parameters(), lr=5e-5)

    best_val_loss = float('inf')
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['input_ids'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['input_ids'].to(device)
                }
                outputs = model(**inputs)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f"mlm_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best MLM model to '{checkpoint_path}'")

    model.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    print("Masked LM Model and tokenizer saved.")
    return model, tokenizer, history

##############################################################################
#                        Contrastive Learning (Siamese)
###############################################################################

class SiameseEncoder(nn.Module):
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2", output_dim=256):
        super(SiameseEncoder, self).__init__()
        self.base_transformer = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.base_transformer.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = self.fc(cls_token)
        return embeddings

def compute_distance(emb1, emb2):
    return torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1, keepdim=True))

def contrastive_loss(margin=1.0):
    def loss_fn(y_true, y_pred):
        y_true = y_true.float()
        d = y_pred
        pos_loss = y_true * torch.pow(d, 2)
        neg_loss = (1 - y_true) * torch.pow(torch.clamp(margin - d, min=0.0), 2)
        return torch.mean(pos_loss + neg_loss)
    return loss_fn

class SiameseModel(nn.Module):
    def __init__(self, encoder, margin=1.0):
        super(SiameseModel, self).__init__()
        self.encoder = encoder
        self.margin = margin

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.encoder(input_ids_1, attention_mask_1)
        emb2 = self.encoder(input_ids_2, attention_mask_2)
        distance = compute_distance(emb1, emb2)
        return distance

def create_siamese_model(encoder, margin=1.0):
    model = SiameseModel(encoder, margin)
    return model

def train_contrastive_learning(text_a, text_b, labels, model_name="aubmindlab/bert-base-arabertv2",
                               max_length=128, margin=1.0, epochs=5, batch_size=2, output_dim=256,
                               output_dir="./contrastive_output", device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc_a = tokenizer(text_a, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    enc_b = tokenizer(text_b, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    class ContrastiveDataset(Dataset):
        def __init__(self, enc_a, enc_b, labels):
            self.input_ids_a = enc_a["input_ids"]
            self.attention_mask_a = enc_a["attention_mask"]
            self.input_ids_b = enc_b["input_ids"]
            self.attention_mask_b = enc_b["attention_mask"]
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'input_ids_1': self.input_ids_a[idx],
                'attention_mask_1': self.attention_mask_a[idx],
                'input_ids_2': self.input_ids_b[idx],
                'attention_mask_2': self.attention_mask_b[idx],
                'label': self.labels[idx]
            }

    dataset = ContrastiveDataset(enc_a, enc_b, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    encoder = SiameseEncoder(model_name, output_dim).to(device)
    model = create_siamese_model(encoder, margin).to(device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion = contrastive_loss(margin)

    best_val_loss = float('inf')
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels_batch = batch['label'].to(device)

            distances = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(labels_batch, distances)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_1 = batch['input_ids_1'].to(device)
                attention_mask_1 = batch['attention_mask_1'].to(device)
                input_ids_2 = batch['input_ids_2'].to(device)
                attention_mask_2 = batch['attention_mask_2'].to(device)
                labels_batch = batch['label'].to(device)

                distances = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = criterion(labels_batch, distances)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f"siamese_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best Siamese contrastive model to '{checkpoint_path}'")

    final_model_path = os.path.join(output_dir, "contrastive_siamese_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Siamese contrastive model saved to '{final_model_path}'")
    return model, tokenizer, history


###############################################################################
#                 Reinforcement Learning (Policy Gradient) for Poem Quality
###############################################################################

def generate_sequence_rl(model, tokenizer, prompt, max_length=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = []
    log_probs = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            token_id = m.sample()
            log_prob = m.log_prob(token_id)

            generated_ids.append(token_id.item())
            log_probs.append(log_prob.item())

            input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, log_probs

def calculate_meter_score(poem):
    # Placeholder function: Implement your meter scoring logic here
    return 1.0  # Example: return a dummy score

def calculate_rhyme_score(poem):
    # Placeholder function: Implement your rhyme scoring logic here
    return 1.0  # Example: return a dummy score

def train_poetic_rl(model_name="aubmindlab/aragpt2-base", 
                    initial_prompt="يا ليل الصب متى غده", 
                    episodes=10, 
                    max_length=20,
                    alpha=0.9,
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    baseline = 0.0
    optimizer = Adam(model.parameters(), lr=1e-5)

    for episode in range(episodes):
        model.train()
        poem, log_probs = generate_sequence_rl(model, tokenizer, initial_prompt, max_length, device)
        # Assuming calculate_meter_score and calculate_rhyme_score are defined
        # Implement your evaluation criteria
        r_m = calculate_meter_score(poem)
        r_r = calculate_rhyme_score(poem)
        reward = (r_m + r_r) / 2.0
        advantage = reward - baseline
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).to(device)
        loss = -torch.sum(log_probs_tensor) * advantage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        baseline = alpha * baseline + (1 - alpha) * reward

        print(f"[Episode {episode+1}/{episodes}] Poem: {poem}")
        print(f"Meter={r_m:.2f}, Rhyme={r_r:.2f}, Reward={reward:.2f}, Baseline={baseline:.2f}")

    model.save_pretrained("./poetic_rl_model")
    tokenizer.save_pretrained("./poetic_rl_model")
    print("Poetic RL Model and tokenizer saved.")
    return model, tokenizer

###############################################################################
#                  Example Main Block (for demonstration)
###############################################################################

if __name__ == "__main__":
    """
    EXAMPLE: 
     1) Load ThePoet pipeline
     2) Load your trained AraGPT2 + Tokenizer
     3) (Optional) Load Diffusion model 
     4) Provide a modern prompt 
     5) Generate final poem via ThePoet -> AraGPT2.
    """
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import sys  # Added for graceful exit

    # Import PyTorch-based functions and classes from utils.py
    # Assuming all classes and functions are defined above

    # --------------------------
    # 1) Define Paths
    # --------------------------
    processed_data_path = '../data/processed/processed_taweel_data.csv'
    diffusion_output_dir = '../models/diffusion'
    transformer_output_dir = '../models/transformers'
    poet_output_dir = '../models/thepoet'  # Optional: Directory to save ThePoet outputs or models

    # Create directories if they don't exist
    os.makedirs(diffusion_output_dir, exist_ok=True)
    os.makedirs(transformer_output_dir, exist_ok=True)
    os.makedirs(poet_output_dir, exist_ok=True)

    # --------------------------
    # 2) Load Processed Data
    # --------------------------
    print("Loading processed data...")
    try:
        processed_df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
        print(f"Processed data loaded with {len(processed_df)} records.")
    except FileNotFoundError:
        print(f"Error: The file '{processed_data_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        sys.exit(1)

    # --------------------------
    # 3) Subset Data (Optional)
    # --------------------------
    subset = True  # Set to False to use the full dataset
    if subset:
        print("Using subset for testing...")
        train_df, valid_df = train_test_split(processed_df, test_size=0.2, random_state=42)
        train_subset = train_df.sample(n=100, random_state=42)
        valid_subset = valid_df.sample(n=20, random_state=42)
    else:
        train_df, valid_df = train_test_split(processed_df, test_size=0.2, random_state=42)
        train_subset, valid_subset = train_df, valid_df

    print(f"Training records: {len(train_subset)}; Validation records: {len(valid_subset)}")

    # --------------------------
    # 4) Load ThePoet Pipeline
    # --------------------------
    print("Creating ThePoet pipeline...")
    try:
        poet_pipeline = create_thepoet_pipeline()
        print("ThePoet pipeline created.")
    except Exception as e:
        print(f"An error occurred while creating ThePoet pipeline: {e}")
        sys.exit(1)

    # --------------------------
    # 5) Initialize and Train AraGPT2 Model
    # --------------------------
    print("Training Transformer (AraGPT2) in auto-encoder style for classical poems...")
    transformer_name = "aubmindlab/aragpt2-base"  # AraGPT2 model
    try:
        transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        # Assign `pad_token` if not already set
        if transformer_tokenizer.pad_token is None:
            transformer_tokenizer.pad_token = transformer_tokenizer.eos_token
            # Alternatively, use a new token for padding:
            # transformer_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Assigned EOS token as PAD token for the tokenizer.")
    except Exception as e:
        print(f"Error loading tokenizer '{transformer_name}': {e}")
        sys.exit(1)

    try:
        transformer_model = AraGPT2ForClassicalStyle(
            model_name=transformer_name,
            freeze_layers=0,   # Adjust if you want to freeze lower layers
            dropout_prob=0.1   # Optional: Adjust dropout probability
        )
    except Exception as e:
        print(f"Error initializing AraGPT2 model: {e}")
        sys.exit(1)

    # Fine-tuning AraGPT2 with improved training function
    try:
        trained_transformer, hist = train_aragpt2_for_classical_style(
            df_classical=train_subset,
            tokenizer=transformer_tokenizer,
            model=transformer_model,
            max_length=128,  # Consistent with max_bayt_len
            epochs=10,        # Increased epochs for better training
            batch_size=4, 
            output_dir=transformer_output_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            freeze_layers=0,       # Adjust as needed
            weight_decay=0.01,     # Added weight decay for regularization
            patience=3,            # Early stopping patience
            max_grad_norm=1.0      # Gradient clipping max norm
        )
        print("Transformer (AraGPT2) training complete.")
    except Exception as e:
        print(f"Error during AraGPT2 training: {e}")
        sys.exit(1)

    # --------------------------
    # 6) Save the Transformer Model and Tokenizer
    # --------------------------
    print("Saving the Transformer model and tokenizer using save_pretrained...")
    try:
        trained_transformer.save_pretrained(transformer_output_dir)
        print(f"Transformer model saved to '{transformer_output_dir}'")
    except Exception as e:
        print(f"Error saving the Transformer model: {e}")
        sys.exit(1)
    
    try:
        transformer_tokenizer.save_pretrained(transformer_output_dir)
        print(f"Tokenizer saved to '{transformer_output_dir}'")
    except Exception as e:
        print(f"Error saving the tokenizer: {e}")
        sys.exit(1)

    # --------------------------
    # 7) Plot Training History
    # --------------------------
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(hist['train_loss'], label='Train Loss')
        plt.plot(hist['val_loss'], label='Validation Loss')
        plt.title("AraGPT2 Fine-Tuning Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
        plt.figure(figsize=(10, 4))
        plt.plot(hist['train_perplexity'], label='Train Perplexity')
        plt.plot(hist['val_perplexity'], label='Validation Perplexity')
        plt.title("AraGPT2 Fine-Tuning Training Perplexity")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error while plotting training history: {e}")

    # -------------
    # 3) Diffusion Model
    # -------------
    # If you wish to include the diffusion model, uncomment and adjust the following:
    """
    max_bayt_len = 128  # Ensure consistency
    encoding_dim = 8
    input_shape = (max_bayt_len, encoding_dim)
    diffusion_model_params = {
        'num_transformer_blocks': 4,
        'num_heads': 8,
        'key_dim': 64,
        'ffn_units': 512
    }
    print("Creating diffusion model...")
    diffusion_model = create_diffusion_model(input_shape, diffusion_model_params)
    print("Diffusion model created.")

    # Vectorize training and validation data
    print("Vectorizing training and validation data...")
    def string_with_tashkeel_vectorizer_per_batch(batch_series, max_bayt_len):
        out = []
        for val in batch_series:
            out.append(string_with_tashkeel_vectorizer(val, max_bayt_len))
        return np.stack(out, axis=0)

    X_train_enc = string_with_tashkeel_vectorizer_per_batch(pd.Series(train_subset['text']), max_bayt_len=128)
    Y_train_enc = X_train_enc.copy()
    X_valid_enc = string_with_tashkeel_vectorizer_per_batch(pd.Series(valid_subset['text']), max_bayt_len=128)
    Y_valid_enc = X_valid_enc.copy()

    batch_size = 8
    epochs = 5
    print("Training diffusion model (example)...")
    ckpt_path = os.path.join(diffusion_output_dir, "diff_ckpt")
    # Replace Keras callbacks with PyTorch equivalents or custom implementations
    # For simplicity, omitted here
    diffusion_history = diffusion_model.fit(
        X_train_enc, Y_train_enc,
        validation_data=(X_valid_enc, Y_valid_enc),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es, tb]
    )

    final_diffusion_path = os.path.join(diffusion_output_dir, "diffusion_model_final.pt")
    torch.save(diffusion_model.state_dict(), final_diffusion_path)
    print(f"Final diffusion model saved to '{final_diffusion_path}'")

    # Plot diffusion training
    plt.figure(figsize=(10, 4))
    plt.plot(diffusion_history['loss'], label='Train Loss')
    plt.plot(diffusion_history['val_loss'], label='Val Loss')
    plt.title("Diffusion Model Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(diffusion_history['mae'], label='Train MAE')
    plt.plot(diffusion_history['val_mae'], label='Val MAE')
    plt.title("Diffusion Model MAE")
    plt.legend()
    plt.show()
    """

    # ------------------
    # 4) Provide a Modern Prompt
    # ------------------
    modern_prompt = "يا جمال الزمان ويا نور الأمل"

    # ------------------
    # 5) Generate Final Poem via ThePoet -> AraGPT2
    # ------------------
    print("Generating final classical poem by chaining ThePoet -> AraGPT2...")
    try:
        final_poem = generate_classical_poem_with_thepoet(
            modern_prompt=modern_prompt,
            poet_pipeline=poet_pipeline,
            transformer_model=trained_transformer,
            transformer_tokenizer=transformer_tokenizer,
            # diffusion_model=diffusion_model,  # Uncomment if using diffusion
            max_length=128,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("\n==== Final Chained Poem ====")
        print(final_poem)
        print("================================")
    except Exception as e:
        print(f"Error during poem generation: {e}")