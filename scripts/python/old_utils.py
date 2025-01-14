# =============================================================================
# Diffusion Model Components
# =============================================================================

import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LayerNormalization, Dense, MultiHeadAttention, Add, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import Sequential

def create_diffusion_model(input_shape, model_params):
    """
    Creates a Transformer-based diffusion model for text generation with
    LayerNormalization layers set to 'float32' to ensure compatibility
    with mixed precision training.
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, encoding_dim).
        model_params (dict): Parameters for model architecture.
    
    Returns:
        tf.keras.Model: Compiled diffusion model.
    """
    # Input layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # LayerNormalization with dtype='float32'
    x = LayerNormalization(dtype='float32', name='layer_normalization')(inputs)
    
    # Transformer Blocks
    for i in range(model_params.get('num_transformer_blocks', 2)):
        # Multi-head self-attention
        attention = MultiHeadAttention(
            num_heads=model_params.get('num_heads', 4),
            key_dim=model_params.get('key_dim', 64),
            name=f'multi_head_attention_{i}'
        )(x, x)
        
        # Add & Normalize with dtype='float32'
        x = Add(name=f'add_attention_{i}')([x, attention])
        x = LayerNormalization(dtype='float32', name=f'layer_normalization_{i}_post_attention')(x)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(model_params.get('ffn_units', 256), activation='relu', name=f'dense_ffn_{i}_1'),
            Dense(input_shape[-1], name=f'dense_ffn_{i}_2')
        ], name=f'ffn_{i}')
        
        ffn_output = ffn(x)
        
        # Add & Normalize with dtype='float32'
        x = Add(name=f'add_ffn_{i}')([x, ffn_output])
        x = LayerNormalization(dtype='float32', name=f'layer_normalization_{i}_post_ffn')(x)
    
    # Output layer with Activation set to 'float32'
    outputs = Dense(
        input_shape[-1],
        activation=None,
        name='output_dense'
    )(x)
    
    # Ensure the output is in 'float32'
    outputs = Activation('linear', dtype='float32', name='output_layer')(outputs)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name='DiffusionModel')
    
    # Compile the model with 'adam' optimizer and appropriate loss and metrics
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Display the model summary
    model.summary()
    
    return model

def train_diffusion_model(model, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, output_dir):
    """
    Trains the diffusion model with specified parameters and callbacks.
    
    Args:
        model (tf.keras.Model): The diffusion model to train.
        X_train (np.ndarray or tf.Tensor): Training input data.
        Y_train (np.ndarray or tf.Tensor): Training target data.
        X_valid (np.ndarray or tf.Tensor): Validation input data.
        Y_valid (np.ndarray or tf.Tensor): Validation target data.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        output_dir (str): Directory to save model checkpoints and logs.
    
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure data is in float32
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float32)
    Y_valid = tf.convert_to_tensor(Y_valid, dtype=tf.float32)
    
    # Define callbacks
    checkpoint_path = os.path.join(
        output_dir, 
        "diffusion_model_epoch_{epoch:02d}_val_mae_{val_mae:.4f}.h5"
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor='val_mae', 
        verbose=1,
        save_best_only=True, 
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_mae', 
        patience=5, 
        verbose=1, 
        mode='min',
        restore_best_weights=True
    )
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
    
    # Save the final model in 'float32' to ensure compatibility
    final_model_path = os.path.join(output_dir, "diffusion_model_final.h5")
    model.save(final_model_path, include_optimizer=False)
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
    
    # Load the model with custom_objects if any custom layers or functions are used
    model = load_model(model_path, compile=True)
    print(f"Loaded diffusion model from {model_path}")
    return model

'''

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


'''











#==========================================================================
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

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Bidirectional, LSTM, Input, LayerNormalization, Dropout,
    Embedding, TimeDistributed
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Transformers
from transformers import (
    TFAutoModel,
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
    TFAutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)

# Arabic-specific modules
import pyarabic.araby as araby

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


AR = Arabic()  # Convenience instance

###############################################################################
#                           Data Cleaning and Preprocessing
###############################################################################

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
    Normalize Arabic text by:
    - Standardizing alef variants.
    - Standardizing ta marbuta and other letters.
    - Removing diacritics except shadda and sukun.
    - Removing tatweel (ـ).
    - Handling hamzas appropriately.
    """
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)

    # remove diacritics except shadda/sukun
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
    Calculate the accuracy percentage of how well a verse aligns with the specified meter pattern.
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
    Extracts basic rhyme info from the 'rhyme' column or from the last word of a verse.
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
    Adds columns for verse length, average shatr length, or other length-based features.
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
            base = char[0]
            # char[1] presumably is AR.shadda
            # char[2] is e.g., AR.fatha
            factoredString += base + AR.sukun + base + char[2]
    return factoredString

def get_alphabet_tashkeel_combination():
    base_letters = AR.alphabet + [' ']
    combos = [''] + base_letters[:]
    for letter in base_letters:
        for haraka in [AR.fatha, AR.damma, AR.kasra, AR.sukun]:
            combos.append(letter + haraka)
    return combos

lettersTashkeelCombination = get_alphabet_tashkeel_combination()
from itertools import product
encoding_combination = [list(i) for i in product([0, 1], repeat=8)]

def string_with_tashkeel_vectorizer(string, padding_length):
    factored_str = factor_shadda_tanwin(string)
    tokens = separate_token_with_diacritics(factored_str)
    representation = []
    for tok in tokens:
        if tok in lettersTashkeelCombination:
            idx = lettersTashkeelCombination.index(tok)
            representation.append(encoding_combination[idx])
        else:
            representation.append([0]*8)
    extra = padding_length - len(representation)
    for _ in range(extra):
        representation.append([0]*8)
    return np.array(representation, dtype=np.int32)

def string_with_tashkeel_vectorizer_per_batch(batch_series, max_bayt_len):
    out = []
    for val in batch_series:
        out.append(string_with_tashkeel_vectorizer(val, max_bayt_len))
    return np.stack(out, axis=0)

def oneHot_per_sample(string, padding_length):
    cleanedString = factor_shadda_tanwin(string)
    charCleanedString = separate_token_with_diacritics(cleanedString)
    encodedString = np.zeros((padding_length, len(lettersTashkeelCombination)), dtype=np.int32)
    letter = 0
    for char in charCleanedString:
        if char in lettersTashkeelCombination:
            one_index = lettersTashkeelCombination.index(char)
            encodedString[letter][one_index] = 1
        letter +=1
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
#                      Transformer Auto-Encoding for Classical Style
###############################################################################

def create_transformer_for_classical_style(model_name='t5-small', max_length=128):
    """
    Creates a Transformer-based Seq2Seq model for 'auto-encoding' on classical poems.
    Uses T5 architecture.
    
    Args:
        model_name (str): Pretrained model name compatible with Seq2Seq.
        max_length (int): Maximum sequence length.
    
    Returns:
        tf.keras.Model: Compiled Transformer model.
    """
    from transformers import TFAutoModelForSeq2SeqLM
    
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train_transformer_for_classical_style(df_classical, tokenizer, model,
                                          max_length=128, epochs=5, batch_size=4,
                                          output_dir='./transformer_output'):
    """
    Trains the Transformer model in a self-supervised manner on classical poems.
    Input=verse, Output=the same verse (auto-encoder style).
    """
    os.makedirs(output_dir, exist_ok=True)
    texts = df_classical['text'].tolist()

    # Prepare data
    enc = tokenizer(texts, max_length=max_length,
                    padding='max_length', truncation=True, return_tensors='np')
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    labels = input_ids.copy()  # auto-encoder => same input as label

    # Split
    train_ids, val_ids, train_mask, val_mask, y_train, y_val = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_ids, 'attention_mask': train_mask},
        y_train
    )).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': val_ids, 'attention_mask': val_mask},
        y_val
    )).batch(batch_size)

    ckpt_path = os.path.join(output_dir, "transformer_ckpt.h5")
    ckpt = ModelCheckpoint(ckpt_path, save_best_only=True,
                           monitor='val_accuracy', mode='max', verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=3,
                       restore_best_weights=True, mode='max')
    tb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[ckpt, es, tb]
    )

    final_model_path = os.path.join(output_dir, "transformer_classical_final.h5")
    model.save_weights(final_model_path)
    print(f"[INFO] Transformer final weights saved to {final_model_path}")

    return model, history

def inference_convert_modern_to_classical(modern_verse, tokenizer, model, max_length=128):
    """
    Given a modern poem, feed it to the transformer model trained in an auto-encoder style
    on classical poems. 
    """
    enc = tokenizer.encode_plus(
        modern_verse,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

###############################################################################
#                  Diffusion Model for Representation Refinement
###############################################################################

def create_diffusion_model(input_shape, model_params):
    from tensorflow.keras.layers import MultiHeadAttention, Add
    inputs = Input(shape=input_shape, name='diffusion_input')
    x = LayerNormalization()(inputs)

    for i in range(model_params.get('num_transformer_blocks', 2)):
        attention = MultiHeadAttention(
            num_heads=model_params.get('num_heads', 4),
            key_dim=model_params.get('key_dim', 64)
        )(x, x)
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        ffn = Sequential([
            Dense(model_params.get('ffn_units', 256), activation='relu'),
            Dense(input_shape[-1])
        ])
        ffn_out = ffn(x)
        x = Add()([x, ffn_out])
        x = LayerNormalization()(x)

    outputs = Dense(input_shape[-1], activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs, name='DiffusionModel')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_diffusion_for_classical_style(df_classical, diffusion_model, tokenizer,
                                        max_length=128, epochs=5, batch_size=4,
                                        output_dir='./diffusion_output'):
    """
    Trains the diffusion model in a self-supervised manner on classical poems.
    We do a dummy approach: feed token IDs as if they are 'embeddings' and
    train the model to reconstruct them.
    """
    os.makedirs(output_dir, exist_ok=True)

    texts = df_classical['text'].tolist()
    enc = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='np')
    X = enc['input_ids']
    Y = X.copy()

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    ckpt_path = os.path.join(output_dir, "diff_ckpt.h5")
    ckpt = ModelCheckpoint(ckpt_path, save_best_only=True,
                           monitor='val_mae', mode='min', verbose=1)
    es = EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True, mode='min')
    tb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))

    history = diffusion_model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es, tb]
    )
    final_path = os.path.join(output_dir, "diffusion_final.h5")
    diffusion_model.save_weights(final_path)
    print(f"[INFO] Diffusion final weights saved to {final_path}")
    return diffusion_model, history

###############################################################################
#                             Evaluation Metrics
###############################################################################

def calculate_meter_score(poem):
    """
    A placeholder meter scoring approach.
    """
    syllables = len(re.findall(r'[اوي]', poem))
    expected_syllables = 10
    return min(syllables / expected_syllables, 1.0)

def calculate_rhyme_score(poem):
    """
    A placeholder rhyme scoring approach.
    """
    lines = poem.strip().split('\n')
    if len(lines) < 2:
        return 0.0
    last_words = [line.strip().split()[-1] for line in lines if line.strip()]
    rhymes = [word[-2:] for word in last_words]
    expected_rhyme = rhymes[0]
    correct = sum(1 for r in rhymes if r == expected_rhyme)
    return correct / len(rhymes)

def compare_with_baselines(results):
    """
    Stub for comparing with baseline scores.
    """
    for result in results:
        result['baseline_meter_score'] = 0.75
        result['baseline_rhyme_score'] = 0.70

###############################################################################
#                           Generation Pipeline
###############################################################################

def generate_classical_poem(modern_poem, transformer_model, tokenizer, diffusion_model,
                            max_length=128):
    """
    1) Use the transformer to convert modern → classical (rough).
    2) Convert that text to embeddings, feed to diffusion model for refinement.
    3) Decode back to text.
    """
    # Step 1
    classical_draft = inference_convert_modern_to_classical(
        modern_verse=modern_poem,
        tokenizer=tokenizer,
        model=transformer_model,
        max_length=max_length
    )
    # Step 2
    enc = tokenizer(
        classical_draft,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    draft_ids = enc['input_ids']  # shape: (1, max_length)
    refined_ids = diffusion_model.predict(draft_ids)
    # Step 3
    refined_ids = np.rint(refined_ids).astype(int).clip(min=0)
    refined_poem = tokenizer.decode(refined_ids[0], skip_special_tokens=True)
    return refined_poem

###############################################################################
#                     Sequence-based Batch Generator
###############################################################################

class ShaarSequence(Sequence):
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
#                        Denoising Auto-Encoder (DAE)
###############################################################################

def create_denoising_autoencoder(vocab_size, embedding_dim=128, latent_dim=256, max_length=128):
    """
    A simple seq2seq denoising auto-encoder for Arabic text.
    """
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(max_length,), name="encoder_input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    
    # Decoder
    decoder_inputs = tf.keras.Input(shape=(max_length,), name="decoder_input")
    dec_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_embed, initial_state=[state_h, state_c])
    decoder_dense = TimeDistributed(Dense(vocab_size, activation="softmax"))
    final_outputs = decoder_dense(decoder_outputs)
    
    dae_model = Model([encoder_inputs, decoder_inputs], final_outputs, name="denoising_autoencoder")
    dae_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return dae_model

def create_noise(input_text, drop_prob=0.2):
    """
    Randomly removes characters in the string to simulate corruption.
    """
    output = []
    for ch in input_text:
        if np.random.rand() < drop_prob:
            continue
        else:
            output.append(ch)
    return "".join(output)

def train_denoising_autoencoder(texts, tokenizer, max_length=128, test_size=0.2, 
                                embedding_dim=128, latent_dim=256,
                                epochs=5, batch_size=4, output_dir="./dae_output"):
    """
    1) Create noisy versions of the input text.
    2) Tokenize both noisy and clean text.
    3) Train the autoencoder to reconstruct clean text from noisy input.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_noisy = [create_noise(t) for t in texts]
    Y_clean = texts

    enc_noisy = tokenizer(
        X_noisy, max_length=max_length, padding="max_length", truncation=True, return_tensors="np"
    )
    enc_clean = tokenizer(
        Y_clean, max_length=max_length, padding="max_length", truncation=True, return_tensors="np"
    )
    X_ids = enc_noisy["input_ids"]
    Y_ids = enc_clean["input_ids"]

    X_train, X_val, Y_train, Y_val = train_test_split(X_ids, Y_ids, test_size=test_size, random_state=42)

    vocab_size = tokenizer.vocab_size
    dae_model = create_denoising_autoencoder(vocab_size, embedding_dim, latent_dim, max_length)

    # For seq2seq: typically pass X as encoder input, shift X by 1 for decoder input
    # Simplistically, we can pass the same input to the decoder:
    # (X_noisy => encoder, X_noisy => decoder, label => Y_clean)
    # Real practice might shift or add special tokens.

    ckpt_path = os.path.join(output_dir, "dae_ckpt.h5")
    ckpt = ModelCheckpoint(ckpt_path, save_best_only=True,
                           monitor='val_accuracy', mode='max', verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max')
    tb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))

    history = dae_model.fit(
        [X_train, X_train], Y_train,
        validation_data=([X_val, X_val], Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es, tb]
    )

    final_model_path = os.path.join(output_dir, "denoising_autoencoder_final.h5")
    dae_model.save(final_model_path)
    print(f"DAE model saved to {final_model_path}")
    return dae_model, history

###############################################################################
#                        Masked Language Modeling (MLM)
###############################################################################

def train_masked_language_model(texts, model_name="aubmindlab/bert-base-arabertv2",
                                max_length=128, epochs=3, batch_size=2, output_dir="./mlm_output"):
    """
    Fine-tunes a Masked LM model on the given Arabic texts.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForMaskedLM.from_pretrained(model_name)

    encodings = tokenizer(
        texts, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length", 
        return_tensors="np"
    )
    input_ids = encodings["input_ids"]

    train_ids, val_ids = train_test_split(input_ids, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ids).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_ids).batch(batch_size)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    def gen_train():
        for batch in train_dataset:
            features = data_collator.torch_call(batch)
            yield ({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]},
                   features["labels"])

    def gen_val():
        for batch in val_dataset:
            features = data_collator.torch_call(batch)
            yield ({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]},
                   features["labels"])

    steps_per_epoch = len(train_ids) // batch_size
    val_steps = len(val_ids) // batch_size

    history = model.fit(
        x=gen_train(),
        validation_data=gen_val(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps
    )

    model.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    print("Masked LM Model and tokenizer saved.")
    return model, tokenizer, history

###############################################################################
#                        Contrastive Learning (Siamese)
###############################################################################

def create_siamese_encoder(model_name="aubmindlab/bert-base-arabertv2", output_dim=256):
    """
    Creates a model that outputs embeddings from a BERT-like transformer.
    """
    base_transformer = TFAutoModel.from_pretrained(model_name)
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    outputs = base_transformer(input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0, :]
    dense = Dense(output_dim, activation=None)(cls_token)
    model = Model([input_ids, attention_mask], dense)
    return model

def compute_distance(emb1, emb2):
    return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis=1, keepdims=True))

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        d = y_pred
        pos_loss = y_true * tf.square(d)
        neg_loss = (1 - y_true) * tf.square(tf.maximum(margin - d, 0))
        return tf.reduce_mean(pos_loss + neg_loss)
    return loss

def create_siamese_model(encoder, margin=1.0):
    input_ids_1 = Input(shape=(None,), dtype=tf.int32, name="input_ids_1")
    attn_mask_1 = Input(shape=(None,), dtype=tf.int32, name="attn_mask_1")
    input_ids_2 = Input(shape=(None,), dtype=tf.int32, name="input_ids_2")
    attn_mask_2 = Input(shape=(None,), dtype=tf.int32, name="attn_mask_2")

    emb1 = encoder([input_ids_1, attn_mask_1])
    emb2 = encoder([input_ids_2, attn_mask_2])
    distance = tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis=1, keepdims=True))

    siamese = Model(
        inputs=[input_ids_1, attn_mask_1, input_ids_2, attn_mask_2],
        outputs=distance
    )
    siamese.compile(optimizer=Adam(1e-5), loss=contrastive_loss(margin))
    return siamese

def train_contrastive_learning(text_a, text_b, labels, model_name="aubmindlab/bert-base-arabertv2",
                               max_length=128, margin=1.0, epochs=5, batch_size=2, output_dim=256,
                               output_dir="./contrastive_output"):
    """
    text_a, text_b: parallel lists of strings
    labels: 1 if they are similar style, 0 otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc_a = tokenizer(text_a, max_length=max_length, truncation=True, padding="max_length", return_tensors="np")
    enc_b = tokenizer(text_b, max_length=max_length, truncation=True, padding="max_length", return_tensors="np")

    encoder = create_siamese_encoder(model_name, output_dim=output_dim)
    siamese = create_siamese_model(encoder, margin=margin)

    history = siamese.fit(
        {
            "input_ids_1": enc_a["input_ids"],
            "attn_mask_1": enc_a["attention_mask"],
            "input_ids_2": enc_b["input_ids"],
            "attn_mask_2": enc_b["attention_mask"]
        },
        np.array(labels),
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )
    final_model_path = os.path.join(output_dir, "contrastive_siamese_model.h5")
    siamese.save(final_model_path)
    print(f"Siamese contrastive model saved to {final_model_path}")
    return siamese, tokenizer, history

###############################################################################
#                 Reinforcement Learning (Policy Gradient) for Poem Quality
###############################################################################

def generate_sequence_rl(model, tokenizer, prompt, max_length=50):
    """
    Generates text step by step from a causal model, collecting log probabilities.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    generated_ids = []
    log_probs = []

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        probs = tf.nn.softmax(next_token_logits, axis=-1)
        token_id = tf.squeeze(tf.random.categorical(tf.math.log(probs), num_samples=1), axis=-1)
        token_prob = tf.reduce_sum(probs * tf.one_hot(token_id, probs.shape[-1]), axis=-1)
        log_prob = tf.math.log(token_prob + 1e-10)

        generated_ids.append(token_id.numpy()[0])
        log_probs.append(log_prob.numpy()[0])

        input_ids = tf.concat([input_ids, tf.expand_dims(token_id, 0)], axis=1)
        # Stop if we see an EOS or special token:
        # if token_id.numpy()[0] == tokenizer.eos_token_id: 
        #     break
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, log_probs

def train_poetic_rl(model_name="aubmindlab/aragpt2-base", 
                    initial_prompt="يا ليل الصب متى غده", 
                    episodes=10, 
                    max_length=20,
                    alpha=0.9):
    """
    Minimal policy gradient approach:
      - Generate poem
      - Calculate reward = meter_score + rhyme_score
      - loss = -log_prob * advantage
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForMaskedLM.from_pretrained(model_name)  # or a real GPT if you have it
    baseline = 0.0
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    for episode in range(episodes):
        with tf.GradientTape() as tape:
            poem, log_probs = generate_sequence_rl(model, tokenizer, initial_prompt, max_length)
            r_m = calculate_meter_score(poem)
            r_r = calculate_rhyme_score(poem)
            reward = (r_m + r_r) / 2.0
            advantage = reward - baseline
            log_probs_tf = tf.constant(log_probs, dtype=tf.float32)
            loss = -tf.reduce_sum(log_probs_tf) * advantage

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        baseline = alpha * baseline + (1 - alpha) * reward

        print(f"[Episode {episode+1}/{episodes}] Poem: {poem}")
        print(f"Meter={r_m:.2f}, Rhyme={r_r:.2f}, Reward={reward:.2f}, Baseline={baseline:.2f}")

    model.save_pretrained("./poetic_rl_model")
    tokenizer.save_pretrained("./poetic_rl_model")
    return model, tokenizer

