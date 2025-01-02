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
    Embedding, TimeDistributed, MultiHeadAttention, Add, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
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
    DataCollatorForLanguageModeling,
    pipeline,
    AutoModelForCausalLM,
    TFAutoModelForCausalLM
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

def create_aragpt2_for_classical_style(model_name='aubmindlab/aragpt2-base', max_length=128, freeze_layers=0):
    """
    Creates an AraGPT2-based Causal Language Model for refining classical poems.

    Args:
        model_name (str): Pretrained AraGPT2 model name.
        max_length (int): Maximum sequence length.
        freeze_layers (int): Number of lower layers to freeze. Default is 0 (no freezing).

    Returns:
        model: Compiled AraGPT2 model.
    """
    from transformers import TFAutoModelForCausalLM
    import tensorflow_addons as tfa
    from tensorflow.keras.losses import SparseCategoricalCrossentropy

    # Load the pre-trained AraGPT2 model
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    
    # Optionally freeze lower layers to prevent overfitting
    if freeze_layers > 0:
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False
        print(f"Frozen the first {freeze_layers} layers of the AraGPT2 model.")
    
    # Define the optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(
        learning_rate=5e-5,         # Reduced learning rate for finer updates
        weight_decay=1e-2,          # Weight decay coefficient
        epsilon=1e-08,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Compile the model with label smoothing
    loss = SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def train_aragpt2_for_classical_style(df_classical, tokenizer, model,
                                      max_length=128, epochs=10, batch_size=4,
                                      output_dir='./transformer_output'):
    """
    Trains the AraGPT2 model for refining classical poems.

    Args:
        df_classical (pd.DataFrame): DataFrame containing classical poems with a 'text' column.
        tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        model (TFAutoModelForCausalLM): Compiled AraGPT2 model.
        max_length (int): Maximum sequence length.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        output_dir (str): Directory to save training logs and checkpoints.

    Returns:
        model: Trained AraGPT2 model.
        history: Training history object.
    """
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

    # Prepare the data
    texts = df_classical['text'].tolist()
    enc = tokenizer(
        texts, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='tf'
    )
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    labels = tf.concat([input_ids[:, 1:], tf.fill([input_ids.shape[0], 1], tokenizer.eos_token_id)], axis=1)

    # Split the data into training and validation sets
    train_ids, val_ids, train_mask, val_mask, y_train, y_val = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": train_ids, "attention_mask": train_mask},
        y_train
    )).shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": val_ids, "attention_mask": val_mask},
        y_val
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define callbacks
    es = EarlyStopping(
        monitor='val_loss',             # Monitor validation loss
        patience=5,                     # Increased patience
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    tb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[es, reduce_lr, tb]
    )
    
    return model, history


def inference_convert_classical(classical_verse, tokenizer, model, max_length=128):
    """
    Converts a diacritized classical verse using the AraGPT2 model.

    Args:
        classical_verse (str): Diacritized classical Arabic verse.
        tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        model (TFAutoModelForCausalLM): Trained AraGPT2 model.
        max_length (int): Maximum sequence length.

    Returns:
        str: Refined classical verse.
    """
    encodings = tokenizer(
        classical_verse,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Generate output using beam search for better quality
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,               # Increased number of beams for diversity
        early_stopping=True,
        no_repeat_ngram_size=2,    # Prevent repeating n-grams
        temperature=1.0            # Adjust temperature as needed
    )

    # Decode the output
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



def inference_convert_classical(classical_verse, tokenizer, model, max_length=128):
    """
    Converts a diacritized classical verse using the AraGPT2 model.
    Args:
        classical_verse (str): Diacritized classical Arabic verse.
        tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        model (TFAutoModelForCausalLM): Trained AraGPT2 model.
        max_length (int): Maximum sequence length.
    Returns:
        str: Refined classical verse.
    """
    encodings = tokenizer(
        classical_verse,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Generate output using beam search for better quality
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,               # Increased number of beams for diversity
        early_stopping=True,
        no_repeat_ngram_size=2,    # Prevent repeating n-grams
        temperature=1.0            # Adjust temperature as needed
    )

    # Decode the output
    refined_verse = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return refined_verse


def create_thepoet_pipeline():
    """
    Creates a text-generation pipeline using the mabaji/thepoet model 
    for generating Arabic poetry.
    """
    poet_tokenizer = AutoTokenizer.from_pretrained("mabaji/thepoet")
    poet_model = AutoModelForCausalLM.from_pretrained("mabaji/thepoet")
    poet_pipeline = pipeline(
        "text-generation", 
        model=poet_model, 
        tokenizer=poet_tokenizer
    )
    return poet_pipeline


def generate_rough_poem_with_thepoet(prompt, poet_pipeline, max_length=50, num_return_sequences=1, max_attempts=5):
    """
    Generates a rough classical Arabic poem based on the given prompt
    using the mabaji/thepoet pipeline. Ensures the following format:
        - First verse matches the input.
        - Each subsequent verse has two halves separated by '-'.
        - Verses are separated by '.'.
    Regenerates if the format is not met.
    """
    attempt = 0
    while attempt < max_attempts:
        # Generate poems using ThePoet
        results = poet_pipeline(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,   # sampling for creativity
            temperature=0.8   # tweak as desired
        )
        generated_texts = [r["generated_text"] for r in results]
        print("Generated rough poems from ThePoet:")
        for idx, text in enumerate(generated_texts, 1):
            print(f"{idx}: {text}")
        
        # Check each generated poem for the required format
        valid_poems = []
        for text in generated_texts:
            # Split the poem into verses based on '.'
            verses = text.split('.')
            verses = [verse.strip() for verse in verses if verse.strip()]
            
            # Ensure the first verse matches the input
            if not verses or verses[0] != prompt:
                continue
            
            # Ensure subsequent verses have two halves separated by '-'
            all_verses_valid = True
            for verse in verses[1:]:
                if '-' not in verse:
                    all_verses_valid = False
                    break
            
            # If the format is valid, add the poem to valid_poems
            if all_verses_valid:
                valid_poems.append(text)
        
        if valid_poems:
            # If at least one valid poem is found, return it
            return valid_poems[0]  # Return the first valid poem
        
        print(f"Attempt {attempt+1}: Generated poems do not match the required format. Regenerating...")
        attempt += 1
    
    raise ValueError(f"Failed to generate a poem with the required format after {max_attempts} attempts.")


def generate_classical_poem_with_thepoet(
    modern_prompt,
    poet_pipeline,
    transformer_model,
    transformer_tokenizer,
    # diffusion_model,  # Removed as we're skipping Diffusion
    max_length=128
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
        transformer_model (TFAutoModelForCausalLM): Fine-tuned AraGPT2 model.
        transformer_tokenizer (AutoTokenizer): Tokenizer for AraGPT2.
        diffusion_model (tf.keras.Model, optional): Diffusion model for further refinement.
        max_length (int): Maximum sequence length for models.

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
            mid_point = len(words) // 2
            verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
            print(f"Verse split into halves: {verse}")

        # Step 4a: Diacritize
        diacritized_verse = add_tashkeel_with_java(verse)
        print(f"Diacritized Verse: {diacritized_verse}")

        # Step 4b: Ensure formatting post-diacritization
        if '   ' not in diacritized_verse:
            words = diacritized_verse.split()
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
            max_length=max_length
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
    encoder_inputs = tf.keras.Input(shape=(max_length,), name="encoder_input")
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    
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
    os.makedirs(output_dir, exist_ok=True)
    X_noisy = [create_noise(t) for t in texts]
    Y_clean = texts

    enc_noisy = tokenizer(X_noisy, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")
    enc_clean = tokenizer(Y_clean, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")
    X_ids = enc_noisy["input_ids"]
    Y_ids = enc_clean["input_ids"]

    X_train, X_val, Y_train, Y_val = train_test_split(X_ids, Y_ids, test_size=test_size, random_state=42)
    vocab_size = tokenizer.vocab_size

    dae_model = create_denoising_autoencoder(vocab_size, embedding_dim, latent_dim, max_length)

    ckpt_path = os.path.join(output_dir, "dae_ckpt.h5")
    ckpt = ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
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
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForMaskedLM.from_pretrained(model_name)

    encodings = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="np")
    input_ids = encodings["input_ids"]

    train_ids, val_ids = train_test_split(input_ids, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ids).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_ids).batch(batch_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf"
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    def gen_train():
        for batch in train_dataset:
            features = data_collator(batch)
            yield ({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]},
                   features["labels"])

    def gen_val():
        for batch in val_dataset:
            features = data_collator(batch)
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

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, log_probs

def train_poetic_rl(model_name="aubmindlab/aragpt2-base", 
                    initial_prompt="يا ليل الصب متى غده", 
                    episodes=10, 
                    max_length=20,
                    alpha=0.9):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
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
    # Define paths
    processed_data_path = '../data/processed/processed_taweel_data.csv'
    diffusion_output_dir = '../models/diffusion'
    transformer_output_dir = '../models/transformers'
    poet_output_dir = '../models/thepoet'  # Optional: Directory to save ThePoet outputs or models

    os.makedirs(diffusion_output_dir, exist_ok=True)
    os.makedirs(transformer_output_dir, exist_ok=True)
    os.makedirs(poet_output_dir, exist_ok=True)

    # Load processed data
    print("Loading processed data...")
    processed_df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
    print(f"Processed data loaded with {len(processed_df)} records.")

    # Optional: subset for quick tests
    subset = True
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
    # 1) Load ThePoet Pipeline
    # --------------------------
    print("Creating ThePoet pipeline...")
    poet_pipeline = create_thepoet_pipeline()
    print("ThePoet pipeline created.")

    # --------------------------
    # 2) Transformer (AraGPT2) Model Training
    # --------------------------
    print("Training Transformer (AraGPT2) in auto-encoder style for classical poems...")
    transformer_name = "aubmindlab/aragpt2-base"  # AraGPT2 model
    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    transformer_model = create_aragpt2_for_classical_style(
        model_name=transformer_name,
        max_length=128,  # Consistent with max_bayt_len
        freeze_layers=0   # Adjust if you want to freeze lower layers
    )

    # Fine-tuning AraGPT2
    trained_transformer, hist = train_aragpt2_for_classical_style(
        df_classical=train_subset,
        tokenizer=transformer_tokenizer,
        model=transformer_model,
        max_length=128,  # Consistent with max_bayt_len
        epochs=3, 
        batch_size=2, 
        output_dir=transformer_output_dir
    )
    print("Transformer (AraGPT2) training complete.")

    # Plot training history (optional)
    plt.figure(figsize=(10, 4))
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title("AraGPT2 Fine-Tuning Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title("AraGPT2 Fine-Tuning Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

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
    ckpt = ModelCheckpoint(ckpt_path, save_best_only=True,
                           monitor='val_mae', mode='min', verbose=1)
    es = EarlyStopping(monitor='val_mae', patience=2, restore_best_weights=True, mode='min')
    tb = TensorBoard(log_dir=os.path.join(diffusion_output_dir, 'logs'))

    diffusion_history = diffusion_model.fit(
        X_train_enc, Y_train_enc,
        validation_data=(X_valid_enc, Y_valid_enc),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es, tb]
    )

    final_diffusion_path = os.path.join(diffusion_output_dir, "diffusion_model_final.h5")
    diffusion_model.save_weights(final_diffusion_path)
    print(f"Final diffusion model saved to {final_diffusion_path}")

    # Plot diffusion training
    plt.figure(figsize=(10, 4))
    plt.plot(diffusion_history.history['loss'], label='Train Loss')
    plt.plot(diffusion_history.history['val_loss'], label='Val Loss')
    plt.title("Diffusion Model Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(diffusion_history.history['mae'], label='Train MAE')
    plt.plot(diffusion_history.history['val_mae'], label='Val MAE')
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
    final_poem = generate_classical_poem_with_thepoet(
        modern_prompt=modern_prompt,
        poet_pipeline=poet_pipeline,
        transformer_model=trained_transformer,
        transformer_tokenizer=transformer_tokenizer,
        # diffusion_model=diffusion_model,  # Uncomment if using diffusion
        max_length=128
    )
    print("\n==== Final Chained Poem ====")
    print(final_poem)
    print("================================")
