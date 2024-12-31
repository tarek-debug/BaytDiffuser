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

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, LayerNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Transformers
from transformers import TFAutoModel, AutoTokenizer

# Arabic-specific modules
import pyarabic.araby as araby

# =============================================================================
# Arabic References and Processing
# =============================================================================

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

AR = Arabic()  # Convenience instance
# =============================================================================
# Data Cleaning and Preprocessing
# =============================================================================

def Clean_data(processed_df, max_bayt_len, verse_column_name='text'):
    """
    Cleans and preprocesses the DataFrame containing Arabic poetry.
    
    Steps:
    - Normalize Hamza variations.
    - Remove non-Arabic characters.
    - Factor shadda and tanwin.
    - Limit text length to 'max_bayt_len'.
    
    Args:
        processed_df (pd.DataFrame): DataFrame with Arabic poetry.
        max_bayt_len (int): Maximum allowed length for each poem.
        verse_column_name (str): Column name containing the poetry text.
    
    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame.
    """
    # Remove diacritics and normalize Hamza
    processed_df['text'] = processed_df[verse_column_name].apply(lambda x: araby.normalize_hamza(x))
    
    # Remove non-Arabic characters (retain spaces and line breaks)
    processed_df['text'] = processed_df['text'].apply(lambda x: re.sub(r'[^\u0600-\u06FF\s]', '', x))
    
    # Apply shadda and tanwin factoring
    processed_df['text'] = processed_df['text'].apply(factor_shadda_tanwin)
    
    # Limit text length
    processed_df = processed_df[processed_df['text'].apply(len) <= max_bayt_len]
    
    return processed_df

# =============================================================================
# Additional Feature Extraction
# =============================================================================

def extract_rhyme_info(df):
    """
    Extracts basic rhyme info from the 'rhyme' column or from the last word of a verse.
    Example: 
       - We can store the last two characters of 'rhyme' as a simplistic rhyme scheme.
       - Or combine with 'combined_verse' to see if the last words match across verses.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least a 'rhyme' column or a 'combined_verse' column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a new 'rhyme_info' column.
    """
    if 'rhyme' in df.columns:
        # Basic approach: take the last 2-3 characters of 'rhyme'
        def get_rhyme_suffix(r):
            return r[-2:] if isinstance(r, str) and len(r) >= 2 else r
        
        df['rhyme_info'] = df['rhyme'].apply(get_rhyme_suffix)
    else:
        # If there's no 'rhyme' col, fallback to last word from 'combined_verse'
        if 'combined_verse' in df.columns:
            def get_last_word_rhyme(text):
                lines = text.split('#')
                # Last line
                last_line = lines[-1].strip()
                last_word = last_line.split()[-1] if last_line.split() else ''
                return last_word[-2:] if len(last_word) >= 2 else last_word

            df['rhyme_info'] = df['combined_verse'].apply(get_last_word_rhyme)
        else:
            df['rhyme_info'] = None
    
    return df

def get_verse_length_features(df):
    """
    Adds columns for verse length, average shatr length, or other length-based features.
    
    Args:
        df (pd.DataFrame): DataFrame containing Arabic poetry. 
                          Expects columns like 'الشطر الايمن' and 'الشطر الايسر' or 'combined_verse'.
    
    Returns:
        pd.DataFrame: Updated DataFrame with new length-based columns.
    """
    if 'combined_verse' in df.columns:
        # Count total length of combined_verse
        df['verse_length'] = df['combined_verse'].apply(lambda v: len(v.replace('#', ' ')))
        # Approx average shatr length
        def avg_shatr_len(v):
            parts = v.split('#')
            if len(parts) == 2:
                left_len = len(parts[0].strip())
                right_len = len(parts[1].strip())
                return (left_len + right_len) / 2
            return len(v)
        df['avg_shatr_length'] = df['combined_verse'].apply(avg_shatr_len)
    else:
        # If we have 'الشطر الايمن' and 'الشطر الايسر'
        if 'الشطر الايسر' in df.columns and 'الشطر الايمن' in df.columns:
            def combined_len(row):
                left_len = len(str(row['الشطر الايسر']))
                right_len = len(str(row['الشطر الايمن']))
                return left_len + right_len

            df['verse_length'] = df.apply(combined_len, axis=1)
            df['avg_shatr_length'] = df['verse_length'] / 2
        else:
            # Fallback
            df['verse_length'] = df['text'].apply(len)
            df['avg_shatr_length'] = df['text'].apply(len)
    
    return df

# =============================================================================
# Tashkeel-Related Functions
# =============================================================================

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
            # Skip unknown characters
            i += 1
    return hroof_with_tashkeel

def factor_shadda_tanwin(string):
    """
    Breaks shadda/tanwin combinations into multi-characters for standardization:
      - e.g., 'مٌ' -> 'مُ + نْ'
      - e.g., shadda -> letter + sukun + letter
    """
    # Turn string into [char + diacritic] forms
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
                    # Handle e.g., "ةٌ" -> "تُ + نْ"
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
                # e.g., 'شّ' -> 'شْ + ش'
                factoredString += base + AR.sukun + base
        else:
            # Length=3 means something like "شَّ"
            # "ش" + "شْ" + next diacritic
            base = char[0]
            # char[1] presumably is AR.shadda
            # char[2] is e.g., AR.fatha
            # => 'base + AR.sukun + base + char[2]'
            factoredString += base + AR.sukun + base + char[2]
    return factoredString

def get_alphabet_tashkeel_combination():
    """
    Creates the list of possible letter+haraka combinations.
    E.g., ['', 'ا', 'ب', ..., 'اَ', 'اُ', 'اِ', 'اْ', etc.]
    """
    # Combine base alphabet & harakat
    # Also add ' ' and '' for potential padding/empty
    base_letters = AR.alphabet + [' ']
    combos = [''] + base_letters[:]
    # Append letter+haraka for all letters in base_letters and harakat
    for letter in base_letters:
        for haraka in [AR.fatha, AR.damma, AR.kasra, AR.sukun]:
            combos.append(letter + haraka)
    return combos

lettersTashkeelCombination = get_alphabet_tashkeel_combination()
encoding_combination = [list(i) for i in product([0, 1], repeat=8)]

def string_with_tashkeel_vectorizer(string, padding_length):
    """
    Maps each [char+diacritic] to an 8-bit vector from 'encoding_combination'.
    Pads the sequence to 'padding_length' with zeros.
    
    Returns: np.array of shape (padding_length, 8).
    """
    # Factor out shaddah/tanwin
    factored_str = factor_shadda_tanwin(string)
    tokens = separate_token_with_diacritics(factored_str)

    representation = []
    for tok in tokens:
        # Find index in lettersTashkeelCombination
        if tok in lettersTashkeelCombination:
            idx = lettersTashkeelCombination.index(tok)
            representation.append(encoding_combination[idx])
        else:
            # Unknown char
            representation.append([0]*8)

    # Pad to 'padding_length' with zeros
    extra = padding_length - len(representation)
    for _ in range(extra):
        representation.append([0]*8)
    return np.array(representation, dtype=np.int32)

def string_with_tashkeel_vectorizer_per_batch(batch_series, max_bayt_len):
    """
    Vectorizes a Pandas Series of strings into shape (len(batch), max_bayt_len, 8).
    """
    out = []
    for val in batch_series:
        out.append(string_with_tashkeel_vectorizer(val, max_bayt_len))
    return np.stack(out, axis=0)

def oneHot_per_sample(string, padding_length):
    """
    One-hot encodes each character with diacritics in the string.
    Returns a (padding_length, len(lettersTashkeelCombination)) array.
    """
    cleanedString = factor_shadda_tanwin(string)
    charCleanedString = separate_token_with_diacritics(cleanedString)

    # Initialize a matrix
    encodedString = np.zeros((padding_length, len(lettersTashkeelCombination)), dtype=np.int32)
    
    letter = 0
    for char in charCleanedString:
        if char in lettersTashkeelCombination:
            one_index = lettersTashkeelCombination.index(char)
            encodedString[letter][one_index] = 1
        letter +=1
        if letter >= padding_length:
            break  # Prevent overflow

    return encodedString

def oneHot_per_batch(batch_strings, padding_length):
    """
    One-hot encodes a batch of strings.
    Returns a (batch_size, padding_length, len(lettersTashkeelCombination)) array.
    """
    # Initialize a 3D matrix
    encodedBatch = np.zeros((len(batch_strings), padding_length, len(lettersTashkeelCombination)), dtype=np.int32)

    # Iterate through each string in the batch
    for i, string in enumerate(batch_strings):
        encoded = oneHot_per_sample(string, padding_length)
        encodedBatch[i] = encoded

    return encodedBatch

def check_percentage_tashkeel(string, threshold=0.4):
    """
    Checks if the percentage of alphabetic characters with diacritics meets the threshold.
    Excludes letters where tashkeel is implied from the calculation.
    
    Returns True if percentage >= threshold, else False.
    """
    # Letters where tashkeel is often implied and doesn't need explicit diacritics
    implied_tashkeel_letters = {AR.alef, AR.waw, AR.yeh}
    
    # Filter out implied tashkeel letters from total_chars
    total_chars = sum(1 for c in string if c.isalpha() and c not in implied_tashkeel_letters)
    if total_chars == 0:
        return False

    # Count characters with explicit diacritics (excluding implied tashkeel letters)
    chars_with_diacritics = sum(
        1 for c in string if c not in implied_tashkeel_letters and c in AR.tashkeel
    )
    
    # Calculate percentage
    percentage = chars_with_diacritics / total_chars
    return percentage >= threshold


def save_h5(file_path, dataset_name, dataVar):
    """
    Saves 'dataVar' as dataset in an HDF5 file at 'file_path'.
    """
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(dataset_name, data=dataVar)
    print(f"Saved dataset '{dataset_name}' to '{file_path}'.")

def load_encoder(encoder_path):
    """
    Loads a pickled encoder object (LabelEncoder or OneHotEncoder).
    """
    print(f"Loading encoder from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully.")
    return encoder

def encode_classes_data(categories):
    """
    Encodes categorical labels using LabelEncoder and OneHotEncoder.
    
    Args:
        categories (pd.Series or list): List of category labels.
    
    Returns:
        tuple: (OneHot encoded labels, LabelEncoder instance)
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(categories)
    
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder

def decode_classes(onehot_vec, encoder):
    """
    Decodes a one-hot vector back to its original class label.
    
    Args:
        onehot_vec (np.array): One-hot encoded vector.
        encoder (LabelEncoder): Fitted LabelEncoder.
    
    Returns:
        str: Decoded class label.
    """
    idx = np.argmax(onehot_vec)
    return encoder.inverse_transform([idx])[0]

# =============================================================================
# Diffusion Model Components
# =============================================================================

def create_diffusion_model(input_shape, model_params):
    """
    Creates a Transformer-based diffusion model for text generation.
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, encoding_dim).
        model_params (dict): Parameters for model architecture.
    
    Returns:
        tf.keras.Model: Compiled diffusion model.
    """
    # Example Transformer-based architecture for diffusion
    inputs = Input(shape=input_shape, name='input_layer')
    x = LayerNormalization()(inputs)
    
    for _ in range(model_params.get('num_transformer_blocks', 2)):
        # Multi-head self-attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=model_params.get('num_heads', 4),
            key_dim=model_params.get('key_dim', 64)
        )(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = LayerNormalization()(x)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(model_params.get('ffn_units', 256), activation='relu'),
            Dense(input_shape[-1])
        ])
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = LayerNormalization()(x)
    
    # Output layer
    outputs = Dense(input_shape[-1], activation='linear', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='DiffusionModel')
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.summary()
    return model

def train_diffusion_model(model, train_data, valid_data, epochs, batch_size, output_dir):
    """
    Trains the diffusion model.
    
    Args:
        model (tf.keras.Model): The diffusion model to train.
        train_data (list of dict): Training data entries.
        valid_data (list of dict): Validation data entries.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        output_dir (str): Directory to save model checkpoints.
    
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare training data
    X_train = np.array([entry['text'] for entry in train_data])
    Y_train = X_train.copy()  # For diffusion models, target is often the clean data
    
    # Prepare validation data
    X_valid = np.array([entry['text'] for entry in valid_data])
    Y_valid = X_valid.copy()
    
    # Define callbacks
    checkpoint_path = os.path.join(output_dir, "diffusion_model_epoch_{epoch:02d}_val_mae_{val_mae:.4f}.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mae', verbose=1,
                                 save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_mae', patience=5, verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs"))
    
    callbacks_list = [checkpoint, early_stopping, tensorboard]
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "diffusion_model_final.h5")
    model.save(final_model_path)
    print(f"Final diffusion model saved to {final_model_path}.")
    
    return history

def load_diffusion_model(model_path):
    """
    Loads a trained diffusion model from the specified path.
    
    Args:
        model_path (str): Path to the saved diffusion model (.h5 file).
    
    Returns:
        tf.keras.Model: Loaded diffusion model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Diffusion model not found at {model_path}")
    model = load_model(model_path)
    print(f"Loaded diffusion model from {model_path}")
    return model

# =============================================================================
# Transformer Model Components
# =============================================================================

def create_transformer_model(model_name, max_length):
    """
    Loads a pre-trained transformer model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model (e.g., 'aubmindlab/bert-base-arabertv2').
        max_length (int): Maximum sequence length.
    
    Returns:
        tuple: (Transformer model, Tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = TFAutoModel.from_pretrained(model_name)
    
    # Freeze transformer layers if needed
    # for layer in transformer.layers:
    #     layer.trainable = False
    
    # Add custom layers for your task
    inputs = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    embeddings = transformer(inputs, attention_mask=attention_mask)[0]
    x = Dense(512, activation='relu')(embeddings)
    x = Dropout(0.3)(x)
    outputs = Dense(512, activation='relu')(x)  # Adjust output units as needed
    
    model = Model(inputs=[inputs, attention_mask], outputs=outputs)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    model.summary()
    return model, tokenizer

def train_transformer_model(model, tokenizer, train_data, valid_data, epochs, batch_size, output_dir, max_length):
    """
    Trains the transformer model.
    
    Args:
        model (tf.keras.Model): The transformer model to train.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the transformer.
        train_data (list of dict): Training data entries.
        valid_data (list of dict): Validation data entries.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        output_dir (str): Directory to save model checkpoints.
        max_length (int): Maximum sequence length for tokenization.
    
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tokenize training data
    train_texts = [entry['text'] for entry in train_data]
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length',
                                max_length=max_length, return_tensors='np')
    
    # Tokenize validation data
    valid_texts = [entry['text'] for entry in valid_data]
    valid_encodings = tokenizer(valid_texts, truncation=True, padding='max_length',
                                max_length=max_length, return_tensors='np')
    
    # Define callbacks
    checkpoint_path = os.path.join(output_dir, "transformer_model_epoch_{epoch:02d}_val_mae_{val_mae:.4f}.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mae', verbose=1,
                                 save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_mae', patience=5, verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join(output_dir, "logs"))
    
    callbacks_list = [checkpoint, early_stopping, tensorboard]
    
    # Train the model
    history = model.fit(
        {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
        train_encodings['input_ids'],  # Assuming target is the input for autoencoding
        validation_data=(
            {'input_ids': valid_encodings['input_ids'], 'attention_mask': valid_encodings['attention_mask']},
            valid_encodings['input_ids']
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "transformer_model_final.h5")
    model.save(final_model_path)
    print(f"Final transformer model saved to {final_model_path}.")
    
    return history

def load_transformer_model(model_path, model_name, max_length):
    """
    Loads a trained transformer model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the saved transformer model (.h5 file).
        model_name (str): Name of the transformer model used during training.
        max_length (int): Maximum sequence length used during training.
    
    Returns:
        tuple: (Loaded Transformer model, Tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model not found at {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded transformer model from {model_path}")
    return model, tokenizer

# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_meter_score(poem):
    """
    Calculates the meter score of the poem.
    Placeholder implementation: Counts syllables or matches known meter patterns.
    
    Args:
        poem (str): The generated poem.
    
    Returns:
        float: Meter adherence score.
    """
    # TODO: Implement actual meter calculation logic
    # Example: Simple heuristic based on syllable counts
    syllables = len(re.findall(r'[اوي]', poem))  # Simplistic example
    expected_syllables = 10  # Example expected count per line
    score = min(syllables / expected_syllables, 1.0)
    return score

def calculate_rhyme_score(poem):
    """
    Calculates the rhyme score of the poem.
    Placeholder implementation: Checks if lines end with the same rhyme.
    
    Args:
        poem (str): The generated poem.
    
    Returns:
        float: Rhyme adherence score.
    """
    # TODO: Implement actual rhyme calculation logic
    lines = poem.strip().split('\n')
    if len(lines) < 2:
        return 0.0
    last_words = [line.strip().split()[-1] for line in lines if line.strip()]
    rhymes = [word[-2:] for word in last_words]  # Simplistic rhyme based on last two characters
    expected_rhyme = rhymes[0]
    correct = sum(1 for rhyme in rhymes if rhyme == expected_rhyme)
    score = correct / len(rhymes)
    return score

def compare_with_baselines(results):
    """
    Compares generated poems' metrics with baseline models.
    Placeholder implementation: Adds baseline comparison data to results.
    
    Args:
        results (list of dict): List containing evaluation metrics for each poem.
    
    Returns:
        None
    """
    # TODO: Implement comparison with actual baseline models
    # Example: Append baseline scores
    for result in results:
        result['baseline_meter_score'] = 0.75  # Example baseline score
        result['baseline_rhyme_score'] = 0.70  # Example baseline score

# =============================================================================
# Generation Pipeline
# =============================================================================

def create_generation_model(transformer_model, diffusion_model):
    """
    Creates a pipeline for generating classical poems using Transformer and Diffusion models.
    
    Args:
        transformer_model (tf.keras.Model): Trained transformer model.
        diffusion_model (tf.keras.Model): Trained diffusion model.
    
    Returns:
        function: A function that takes a prompt and generates a classical poem.
    """
    def generate_classical_poem(prompt, tokenizer, transformer_model, diffusion_model, max_length=1000):
        """
        Generates a classical poem based on the input prompt.
        
        Args:
            prompt (str): Modern Arabic poem or input text.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the transformer model.
            transformer_model (tf.keras.Model): Trained transformer model.
            diffusion_model (tf.keras.Model): Trained diffusion model.
            max_length (int): Maximum sequence length.
        
        Returns:
            str: Generated classical poem.
        """
        # Tokenize the prompt
        encoding = tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Generate initial embedding from transformer
        transformer_output = transformer_model.predict(
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )
        
        # Pass through diffusion model to refine
        diffusion_output = diffusion_model.predict(transformer_output)
        
        # Decode the refined embedding back to text
        # Placeholder: Assuming diffusion_output can be directly decoded
        # In practice, you might need a separate decoder or mapping
        # For example purposes, we'll return the prompt itself
        # TODO: Implement proper decoding from diffusion_output to text
        generated_poem = prompt  # Placeholder
        
        return generated_poem
    
    return partial(generate_classical_poem, tokenizer=None, transformer_model=transformer_model, diffusion_model=diffusion_model)

# =============================================================================
# Sequence-based Batch Generator for Large Datasets
# =============================================================================

class ShaarSequence(Sequence):
    """
    Sequence-based batch generator for large datasets,
    using a vectorization function that transforms text -> numeric array.
    """
    def __init__(self, batch_size, bayt_dataset, bhore_dataset,
                 vectorize_fun_batch, max_bayt_len):
        self.batch_size = batch_size
        self.bayt_dataset = bayt_dataset.reset_index(drop=True)
        self.bhore_dataset = bhore_dataset
        self.vectorize_fun_batch = vectorize_fun_batch
        self.max_bayt_len = max_bayt_len

    def __len__(self):
        return int(np.ceil(len(self.bayt_dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min((idx + 1) * self.batch_size, len(self.bayt_dataset))
        batch_text = self.bayt_dataset[start:end]
        batch_y    = self.bhore_dataset[start:end]
        X_enc = self.vectorize_fun_batch(batch_text, self.max_bayt_len)
        return X_enc, batch_y

# =============================================================================
# Logging and Checkpoint Utilities
# =============================================================================

def update_log_file(exp_name, text, epoch_flag=False):
    """
    Updates 'log.txt' with experiment results or epoch counts.
    
    Args:
        exp_name (str): Name of the experiment.
        text (str): Text to update or epoch count.
        epoch_flag (bool): If True, increment epoch count; else, update with text.
    
    Returns:
        bool: True if update is successful, else False.
    """
    def _update_line(line, newtext, ep_flag):
        if ep_flag:
            # Increment epoch number
            prefix, epoch_count = line.split("@")
            epoch_count = str(int(epoch_count) + 1)
            return prefix + "@" + epoch_count
        else:
            # Update the text after comma
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
    """
    Keeps only the best checkpoint (highest validation metric) and removes others.
    
    Args:
        checkpoints_path (str): Directory containing checkpoint files.
    
    Returns:
        None
    """
    try:
        # List all checkpoint files
        models = os.listdir(checkpoints_path)
        # Extract validation metric from filenames
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

        # Sort based on metric (assuming lower is better for MAE)
        metric_files_sorted = sorted(metric_files, key=lambda x: x[1])
        best_file = metric_files_sorted[0][0]  # best one
        print(f"Best checkpoint identified: {best_file}")

        # Remove all other checkpoints except the best one
        for file, _ in metric_files_sorted[1:]:
            os.remove(os.path.join(checkpoints_path, file))
            print(f"Removed checkpoint: {file}")
    except Exception as e:
        print(f"Error in removing non-max checkpoints: {e}")

# =============================================================================
# Evaluation Utilities
# =============================================================================

def recall_precision_f1(confusion_matrix_df):
    """
    Evaluates confusion matrix to produce recall, precision, and F1 score across classes.
    
    Args:
        confusion_matrix_df (pd.DataFrame): Confusion matrix with true labels as index and predicted labels as columns.
    
    Returns:
        tuple: (DataFrame with Recall and Precision per class, F1 Score)
    """
    confusion_matrix_np = confusion_matrix_df.values
    num_classes = confusion_matrix_np.shape[0]
    
    # Calculate per-class recall and precision
    recall_per_class = np.diag(confusion_matrix_np) / np.sum(confusion_matrix_np, axis=1)
    precision_per_class = np.diag(confusion_matrix_np) / np.sum(confusion_matrix_np, axis=0)
    
    # Handle division by zero
    recall_per_class = np.nan_to_num(recall_per_class)
    precision_per_class = np.nan_to_num(precision_per_class)
    
    # Calculate F1 Score
    f1_scores = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    f1_scores = np.nan_to_num(f1_scores)
    f1_score = np.mean(f1_scores)
    
    # Build the DataFrame
    results_df = pd.DataFrame({
        'Recall': recall_per_class,
        'Precision': precision_per_class
    }, index=confusion_matrix_df.index)
    
    return results_df, f1_score

def calculate_meter_score(poem):
    """
    Calculates the meter score of the poem.
    Placeholder implementation: Counts syllables or matches known meter patterns.
    
    Args:
        poem (str): The generated poem.
    
    Returns:
        float: Meter adherence score.
    """
    # TODO: Implement actual meter calculation logic
    # Example: Simple heuristic based on syllable counts
    syllables = len(re.findall(r'[اوي]', poem))  # Simplistic example
    expected_syllables = 10  # Example expected count per line
    score = min(syllables / expected_syllables, 1.0)
    return score

def calculate_rhyme_score(poem):
    """
    Calculates the rhyme score of the poem.
    Placeholder implementation: Checks if lines end with the same rhyme.
    
    Args:
        poem (str): The generated poem.
    
    Returns:
        float: Rhyme adherence score.
    """
    # TODO: Implement actual rhyme calculation logic
    lines = poem.strip().split('\n')
    if len(lines) < 2:
        return 0.0
    last_words = [line.strip().split()[-1] for line in lines if line.strip()]
    rhymes = [word[-2:] for word in last_words]  # Simplistic rhyme based on last two characters
    expected_rhyme = rhymes[0]
    correct = sum(1 for rhyme in rhymes if rhyme == expected_rhyme)
    score = correct / len(rhymes)
    return score

def compare_with_baselines(results):
    """
    Compares generated poems' metrics with baseline models.
    Placeholder implementation: Adds baseline comparison data to results.
    
    Args:
        results (list of dict): List containing evaluation metrics for each poem.
    
    Returns:
        None
    """
    # TODO: Implement comparison with actual baseline models
    # Example: Append baseline scores
    for result in results:
        result['baseline_meter_score'] = 0.75  # Example baseline score
        result['baseline_rhyme_score'] = 0.70  # Example baseline score

# =============================================================================
# Generation Pipeline
# =============================================================================

def generate_classical_poem(prompt, transformer_model, tokenizer, diffusion_model, max_length=1000):
    """
    Generates a classical poem based on the input prompt using Transformer and Diffusion models.
    
    Args:
        prompt (str): Modern Arabic poem or input text.
        transformer_model (tf.keras.Model): Trained transformer model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the transformer model.
        diffusion_model (tf.keras.Model): Trained diffusion model.
        max_length (int): Maximum sequence length.
    
    Returns:
        str: Generated classical poem.
    """
    # Tokenize the prompt
    encoding = tokenizer.encode_plus(
        prompt,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Generate initial embedding from transformer
    transformer_output = transformer_model.predict(
        {'input_ids': input_ids, 'attention_mask': attention_mask}
    )
    
    # Pass through diffusion model to refine
    diffusion_output = diffusion_model.predict(transformer_output)
    
    # Decode the refined embedding back to text
    # Placeholder: Assuming diffusion_output can be directly decoded
    # In practice, you might need a separate decoder or mapping
    # For example purposes, we'll return the prompt itself
    # TODO: Implement proper decoding from diffusion_output to text
    generated_poem = prompt  # Placeholder
    
    return generated_poem

def load_transformer_model(model_path, model_name, max_length):
    """
    Loads a trained transformer model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the saved transformer model (.h5 file).
        model_name (str): Name of the transformer model used during training.
        max_length (int): Maximum sequence length used during training.
    
    Returns:
        tuple: (Loaded Transformer model, Tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model not found at {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    print(f"Loaded transformer model from {model_path}")
    return model, tokenizer

def load_diffusion_model(model_path):
    """
    Loads a trained diffusion model from the specified path.
    
    Args:
        model_path (str): Path to the saved diffusion model (.h5 file).
    
    Returns:
        tf.keras.Model: Loaded diffusion model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Diffusion model not found at {model_path}")
    model = load_model(model_path)
    print(f"Loaded diffusion model from {model_path}")
    return model

# =============================================================================
# End of utils.py
# =============================================================================
