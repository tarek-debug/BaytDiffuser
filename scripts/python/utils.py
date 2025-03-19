# imports.py

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
from tqdm import tqdm  # For progress bars

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
    AutoModelForCausalLM,
    GPT2Tokenizer
)
from arabert import ArabertPreprocessor
from arabert.aragpt2.grover.modeling_gpt2 import GPT2LMHeadModel

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
            base = char[0]
            factoredString += base + AR.sukun + base + char[2]
    return factoredString

def Clean_data(processed_df, max_bayt_len, verse_column_name='text'):
    """
    Cleans and preprocesses a DataFrame containing Arabic poetry.
    """
    processed_df['text'] = processed_df[verse_column_name].apply(lambda x: araby.normalize_hamza(x))
    processed_df['text'] = processed_df['text'].apply(lambda x: re.sub(r'[^\u0600-\u06FF\s]', '', x))
    processed_df['text'] = processed_df['text'].apply(factor_shadda_tanwin)
    processed_df = processed_df[processed_df['text'].apply(len) <= max_bayt_len]
    return processed_df

###############################################################################
#                           Meter Accuracy & Normalization
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
    vowels_short = [AR.fatha, AR.kasra, AR.damma]
    vowels_long = [AR.alef, AR.waw, AR.yeh]
    shadda = AR.shadda
    sukun = AR.sukun

    i = 0
    while i < len(verse):
        char = verse[i]
        next_char = verse[i+1] if i+1 < len(verse) else ''
        if char in AR.alphabet and char not in vowels_long:
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
#                         Additional Feature Extraction
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
        if freeze_layers > 0:
            for name, param in self.model.named_parameters():
                try:
                    layer_num = int(name.split('.')[2]) if len(name.split('.')) > 2 else -1
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                except (IndexError, ValueError):
                    continue
            print(f"Frozen the first {freeze_layers} layers of the AraGPT2 model.")
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.dropout(outputs.logits)
        return outputs.loss, logits
    
    def save_pretrained(self, save_directory):
        if os.path.exists(save_directory):
            if os.path.isfile(save_directory):
                os.remove(save_directory)
                print(f"Removed existing file at '{save_directory}' to create directory.")
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        print(f"Underlying Hugging Face model saved to '{save_directory}'")
        dropout_path = os.path.join(save_directory, 'dropout.pt')
        torch.save(self.dropout.state_dict(), dropout_path)
        print(f"Custom dropout layer state_dict saved to '{dropout_path}'")
    
    @classmethod
    def from_pretrained(cls, save_directory, freeze_layers=0, dropout_prob=0.1):
        model_instance = cls(model_name=save_directory, freeze_layers=freeze_layers, dropout_prob=dropout_prob)
        model_instance.model = AutoModelForCausalLM.from_pretrained(save_directory)
        print(f"Underlying Hugging Face model loaded from '{save_directory}'")
        dropout_path = os.path.join(save_directory, 'dropout.pt')
        if os.path.exists(dropout_path):
            model_instance.dropout.load_state_dict(torch.load(dropout_path, map_location='cpu'))
            print(f"Custom dropout layer state_dict loaded from '{dropout_path}'")
        else:
            print(f"No custom dropout layer found at '{dropout_path}'. Using default dropout.")
        return model_instance

def preprocess_texts(texts, preprocessor):
    """
    Applies ArabertPreprocessor to a list of texts.
    """
    return [preprocessor.preprocess(text) for text in texts]

def embed_tokens_pytorch(input_ids, max_bayt_len, encoding_dim):
    """
    Embeds input IDs by repeating them along the last dimension.
    """
    batch_size, seq_len = input_ids.size()
    if seq_len < max_bayt_len:
        padding = torch.zeros((batch_size, max_bayt_len - seq_len), dtype=input_ids.dtype, device=input_ids.device)
        input_ids_padded = torch.cat([input_ids, padding], dim=1)
    else:
        input_ids_padded = input_ids[:, :max_bayt_len]
    embedded = input_ids_padded.unsqueeze(-1).repeat(1, 1, encoding_dim).float()
    return embedded

def train_aragpt2_for_classical_style(df_classical, tokenizer, model, preprocessor,
                                      max_length=128, epochs=10, batch_size=8,
                                      output_dir='./transformer_output', device='cuda' if torch.cuda.is_available() else 'cpu',
                                      freeze_layers=0, weight_decay=0.01, 
                                      patience=3, max_grad_norm=1.0):
    """
    Trains the AraGPT2 model for refining classical poems.
    """
    os.makedirs(output_dir, exist_ok=True)
    texts = df_classical['text'].tolist()
    preprocessed_texts = preprocess_texts(texts, preprocessor)
    encodings = tokenizer(
        preprocessed_texts, 
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
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_perplexity': [], 'val_perplexity': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss, logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * input_ids_batch.size(0)
        
        avg_train_loss = total_loss / train_size
        train_perplexity = math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                input_ids_batch, attention_mask_batch, labels_batch = [b.to(device) for b in batch]
                loss, logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                val_loss += loss.item() * input_ids_batch.size(0)
        
        avg_val_loss = val_loss / val_size
        val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_perplexity:.2f} | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Perplexity/Train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)
        
        history['train_loss'].append(avg_train_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_loss'].append(avg_val_loss)
        history['val_perplexity'].append(val_perplexity)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = os.path.join(output_dir, f"model_epoch_{epoch}")
            model.save_pretrained(checkpoint_dir)
            print(f"Saved best model to '{checkpoint_dir}'")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    
    writer.close()
    print("Training complete.")
    return model, history

def inference_convert_classical(classical_verse, tokenizer, model, max_length=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Converts a diacritized classical verse using the AraGPT2 model.
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
        generated_ids = model.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=1.0
        )
    
    refined_verse = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return refined_verse

###############################################################################
#            NEW: ThePoet Integration for Initial Generation
###############################################################################

def add_tashkeel_with_java(verse):
    """
    Calls the VerseDiacritizer Java program to diacritize an Arabic verse.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir("../scripts/java")
        command = [
            "java",
            "-cp",
            ".:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/FarasaDiacritizeJar.jar:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/lib/*",
            "VerseDiacritizer",
            verse
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print("Error in diacritization:", result.stderr)
            return verse
    except Exception as e:
        print(f"Exception occurred: {e}")
        return verse
    finally:
        os.chdir(original_cwd)

def create_thepoet_pipeline():
    """
    Creates a text-generation pipeline using the mabaji/thepoet model.
    """
    poet_tokenizer = AutoTokenizer.from_pretrained("mabaji/thepoet")
    poet_model = AutoModelForCausalLM.from_pretrained("mabaji/thepoet")
    poet_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    poet_pipeline_instance = pipeline(
        "text-generation", 
        model=poet_model, 
        tokenizer=poet_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return poet_pipeline_instance

def generate_rough_poem_with_thepoet(prompt, poet_pipeline, max_length=50, num_return_sequences=1, max_attempts=10):
    """
    Generates a rough classical Arabic poem based on the given prompt using ThePoet.
    """
    import re
    verses = []
    attempts = 0
    desired_verse_count = 6
    current_prompt = prompt

    while len(verses) < desired_verse_count and attempts < max_attempts:
        try:
            results = poet_pipeline(
                current_prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=0.8
            )
            generated_texts = [r["generated_text"] for r in results]
            print("\nGenerated rough poems from ThePoet:")
            for idx, text in enumerate(generated_texts, 1):
                print(f"{idx}: {text}")
            for text in generated_texts:
                split_verses = text.split('.')
                split_verses = [v.strip() for v in split_verses if v.strip()]
                for verse in split_verses:
                    if not verses and verse != prompt:
                        print("First verse does not match the prompt. Skipping.")
                        continue
                    if not verses and verse == prompt:
                        verses.append(verse)
                        continue
                    if '   ' not in verse and '-' not in verse:
                        words = verse.split()
                        if len(words) < 2:
                            print(f"Verse too short to split into halves: '{verse}'. Skipping.")
                            continue
                        mid_point = len(words) // 2
                        fixed_verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
                        if not fixed_verse.endswith('.'):
                            fixed_verse += '.'
                        verses.append(fixed_verse)
                    else:
                        fixed_verse = verse.replace('-', '   ') if '-' in verse else verse
                        if not fixed_verse.endswith('.'):
                            fixed_verse += '.'
                        verses.append(fixed_verse)
                    current_prompt = verse
                    if len(verses) >= desired_verse_count:
                        break
            if len(verses) < desired_verse_count:
                attempts += 1
                print(f"Attempt {attempts}: Generated poems do not match the required format. Regenerating...")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            attempts += 1
            continue

    if len(verses) < desired_verse_count and verses:
        print("\nFinalizing the poem by trimming the last incomplete verse if necessary.")
        last_verse = verses[-1]
        if '.' in last_verse:
            trimmed_verse = last_verse.rsplit('.', 1)[0].strip()
            if '   ' in trimmed_verse:
                verses[-1] = trimmed_verse + '.'
                print(f"Trimmed last verse: {verses[-1]}")
            else:
                words = trimmed_verse.split()
                if len(words) < 2:
                    verses[-1] = trimmed_verse + '.'
                    print(f"Trimmed last verse (cannot split further): {verses[-1]}")
                else:
                    mid_point = len(words) // 2
                    fixed_verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:]) + '.'
                    verses[-1] = fixed_verse
                    print(f"Fixed last verse by splitting into halves: {verses[-1]}")
    for i, verse in enumerate(verses):
        if not verse.endswith('.'):
            verses[i] = verse + '.'
            print(f"Added missing period to verse {i+1}: {verses[i]}")
    if len(verses) > desired_verse_count:
        verses = verses[:desired_verse_count]
        print(f"\nTrimmed the poem to the first {desired_verse_count} verses.")
    elif len(verses) < desired_verse_count:
        while len(verses) < desired_verse_count:
            verses.append(".")
            print("Added empty verse to reach six verses.")
    final_poem = '\n'.join(verses)
    print(f"\n==== Final Chained Poem ====\n{final_poem}\n================================")
    return final_poem

###############################################################################
#              Inference Functions for Additional Models
###############################################################################

def inference_dae(text, tokenizer, dae_model, max_length=128, device='cuda'):
    """
    Uses the Denoising Autoencoder to refine the given text.
    """
    dae_model.eval()
    enc = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = dae_model(input_ids, input_ids)
    logits = outputs.view(-1, tokenizer.vocab_size)
    predicted_ids = torch.argmax(logits, dim=1).view(1, -1)
    refined_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return refined_text

def inference_mlm(text, tokenizer, mlm_model, max_length=128, device='cuda'):
    """
    Uses a Masked Language Model to further refine the text.
    """
    mlm_model.eval()
    enc = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    refined_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return refined_text

def inference_diffusion(text, tokenizer, diffusion_model, max_length=128, max_bayt_len=128, encoding_dim=8, device='cuda'):
    """
    Uses the Diffusion model to refine the text.
    """
    diffusion_input_enc = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    diffusion_input_ids = diffusion_input_enc['input_ids'].to(device)
    diffusion_embeddings = embed_tokens_pytorch(diffusion_input_ids, max_bayt_len, encoding_dim).to(device)
    with torch.no_grad():
        outputs = diffusion_model(diffusion_embeddings, labels=diffusion_input_ids)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        refined_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        refined_text = filter_arabic(refined_text)
    return refined_text

###############################################################################
#    Modified: Generate Classical Poem with the Full Model Chain
###############################################################################

def generate_classical_poem_with_thepoet(
    modern_prompt,
    poet_pipeline,
    transformer_model,
    transformer_tokenizer,
    diffusion_model=None,
    dae_model=None,
    mlm_model=None,
    max_length=128,
    max_bayt_len=128,
    encoding_dim=8,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generates a refined classical Arabic poem by chaining:
        1. ThePoet for rough generation,
        2. AraGPT2 for initial refinement,
        3. Denoising Autoencoder (DAE) for further denoising,
        4. Diffusion Model for additional refinement,
        5. Masked Language Model (MLM) for final smoothing.
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
    rough_poem = re.sub(r'[-–—]', '   ', rough_poem)
    rough_poem = re.sub(r'\s+', ' ', rough_poem).strip()
    print(f"\nRough Poem after formatting:\n{rough_poem}")
    verse_delimiters = r'[.!؟]'
    verses = re.split(verse_delimiters, rough_poem)
    verses = [verse.strip() for verse in verses if verse.strip()]
    print(f"\nNumber of verses extracted: {len(verses)}")
    for idx, verse in enumerate(verses, 1):
        print(f"Verse {idx}: {verse}")
    processed_verses = []
    for idx, verse in enumerate(verses, 1):
        print(f"\nProcessing Verse {idx}: {verse}")
        if '   ' not in verse:
            words = verse.split()
            if len(words) < 2:
                print(f"Verse too short to split into halves: '{verse}'. Skipping.")
                continue
            mid_point = len(words) // 2
            verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
            print(f"Verse split into halves: {verse}")
        diacritized_verse = add_tashkeel_with_java(verse)
        print(f"Diacritized Verse: {diacritized_verse}")
        if '   ' not in diacritized_verse:
            words = diacritized_verse.split()
            if len(words) < 2:
                diacritized_verse = diacritized_verse + '.'
                print(f"Diacritized Verse too short. Added period: {diacritized_verse}")
            else:
                mid_point = len(words) // 2
                diacritized_verse = ' '.join(words[:mid_point]) + '   ' + ' '.join(words[mid_point:])
                print(f"Diacritized Verse re-split into halves: {diacritized_verse}")
        else:
            diacritized_verse = re.sub(r'\s+', '   ', diacritized_verse).strip()
            print(f"Diacritized Verse normalized: {diacritized_verse}")
        classical_draft = inference_convert_classical(
            classical_verse=diacritized_verse,
            tokenizer=transformer_tokenizer,
            model=transformer_model,
            max_length=max_length,
            device=device
        )
        print(f"Classical Draft from AraGPT2: {classical_draft}")
        classical_draft_clean = filter_arabic(classical_draft)
        print(f"After filtering non-Arabic characters: {classical_draft_clean}")

        # Sequential Refinement: DAE -> Diffusion -> MLM
        refined_text = classical_draft_clean
        if dae_model is not None:
            refined_text = inference_dae(refined_text, transformer_tokenizer, dae_model, max_length, device)
            print(f"After DAE refinement: {refined_text}")
        if diffusion_model is not None:
            refined_text = inference_diffusion(refined_text, transformer_tokenizer, diffusion_model, max_length, max_bayt_len, encoding_dim, device)
            print(f"After Diffusion refinement: {refined_text}")
        if mlm_model is not None:
            refined_text = inference_mlm(refined_text, transformer_tokenizer, mlm_model, max_length, device)
            print(f"After MLM refinement: {refined_text}")

        if refined_text:
            processed_verses.append(refined_text)
        else:
            print(f"Warning: Final verse {idx} is empty after refinement.")

    final_poem = '\n'.join(processed_verses)
    print(f"\n==== Final Chained Poem ====\n{final_poem}\n================================")
    return final_poem

###############################################################################
#                        Diffusion Model in PyTorch
###############################################################################

class DiffusionDatasetWithLabels(Dataset):
    def __init__(self, embedded_X, input_ids):
        self.X = embedded_X.float()
        self.labels = input_ids.long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, key_dim, ffn_units):
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
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, model_params):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_transformer_blocks = model_params.get('num_transformer_blocks', 2)
        self.num_heads = model_params.get('num_heads', 4)
        self.key_dim = model_params.get('key_dim', 64)
        self.ffn_units = model_params.get('ffn_units', 256)
        self.initial_norm = nn.LayerNorm(input_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(input_dim=input_dim, num_heads=self.num_heads, key_dim=self.key_dim, ffn_units=self.ffn_units)
            for _ in range(self.num_transformer_blocks)
        ])
        self.output_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        x = self.initial_norm(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

def create_diffusion_model_pytorch(input_shape, model_params):
    seq_len, encoding_dim = input_shape
    model = DiffusionModel(input_dim=encoding_dim, model_params=model_params)
    return model

class DiffusionModelWithDecoder(nn.Module):
    def __init__(self, diffusion_model, gpt2_model_name='aubmindlab/aragpt2-base'):
        super(DiffusionModelWithDecoder, self).__init__()
        self.diffusion_model = diffusion_model
        self.decoder = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        diffusion_output_dim = self.diffusion_model.output_layer.out_features
        gpt2_embedding_dim = self.decoder.transformer.wte.embedding_dim
        if diffusion_output_dim != gpt2_embedding_dim:
            self.projection = nn.Linear(diffusion_output_dim, gpt2_embedding_dim)
            print(f"Projection layer added: {diffusion_output_dim} -> {gpt2_embedding_dim}")
        else:
            self.projection = None
            print("No projection layer needed; dimensions match.")
    
    def forward(self, x, labels=None):
        refined_embeddings = self.diffusion_model(x)
        if self.projection:
            refined_embeddings = self.projection(refined_embeddings)
        outputs = self.decoder(inputs_embeds=refined_embeddings, labels=labels)
        return outputs

def train_diffusion_with_gpt2_decoder(
    df_classical, 
    diffusion_model, 
    tokenizer, 
    preprocessor,
    max_length=128, 
    max_bayt_len=128, 
    encoding_dim=8,
    epochs=10, 
    batch_size=8, 
    output_dir='./diffusion_output_with_decoder',
    learning_rate=1e-4, 
    patience=3, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(output_dir, exist_ok=True)
    combined_model = DiffusionModelWithDecoder(diffusion_model).to(device)
    texts = df_classical['text'].tolist()
    preprocessed_texts = preprocess_texts(texts, preprocessor)
    preprocessed_texts = [filter_arabic(text) for text in preprocessed_texts]
    encodings = tokenizer(
        preprocessed_texts, 
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
    embedded_X = embed_tokens_pytorch(input_ids, max_bayt_len, encoding_dim)
    dataset_with_labels = DiffusionDatasetWithLabels(embedded_X, input_ids)
    train_size = int(0.8 * len(dataset_with_labels))
    val_size = len(dataset_with_labels) - train_size
    train_dataset, val_dataset = random_split(dataset_with_labels, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = Adam(combined_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        combined_model.train()
        train_loss = 0.0
        for batch_X, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            batch_X = batch_X.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = combined_model(batch_X, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = train_loss / train_size
        combined_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                batch_X = batch_X.to(device)
                batch_labels = batch_labels.to(device)
                outputs = combined_model(batch_X, labels=batch_labels)
                loss = outputs.loss
                val_loss += loss.item() * batch_X.size(0)
        avg_val_loss = val_loss / val_size
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f"diffusion_model_epoch_{epoch}.pt")
            torch.save(combined_model.state_dict(), checkpoint_path)
            print(f"Saved best model to '{checkpoint_path}'")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    writer.close()
    print("Training complete.")
    return combined_model, history

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
    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        dae_model.train()
        total_loss = 0
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
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        dae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = [b.to(device) for b in batch]
                outputs = dae_model(noisy, noisy)
                logits = outputs.view(-1, vocab_size)
                clean = clean.view(-1)
                loss = criterion(logits, clean)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
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
            return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}
    train_dataset = MLMDataset(train_ids, train_mask)
    val_dataset = MLMDataset(val_ids, val_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f"mlm_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best MLM model to '{checkpoint_path}'")
    model.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "masked_lm_model"))
    print("Masked LM Model and tokenizer saved.")
    return model, tokenizer, history

###############################################################################
#                 Reinforcement Learning for Poem Quality
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
    # Placeholder scoring logic
    return 1.0

def calculate_rhyme_score(poem):
    # Placeholder scoring logic
    return 1.0

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
#                                FULL MAIN
###############################################################################

if __name__ == "__main__":
    import sys
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # --------------------------
    # 1) Define Paths & Create Directories
    # --------------------------
    processed_data_path = '../data/processed/processed_taweel_data.csv'
    transformer_output_dir = '../models/transformers'
    diffusion_output_dir = '../models/diffusion'
    dae_output_dir = '../models/dae'
    mlm_output_dir = '../models/mlm'
    rl_output_dir = '../models/rl'
    poet_output_dir = '../models/thepoet'
    for d in [transformer_output_dir, diffusion_output_dir, dae_output_dir, mlm_output_dir, rl_output_dir, poet_output_dir]:
        os.makedirs(d, exist_ok=True)

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
    # 3) Subset Data for Testing
    # --------------------------
    subset = True
    if subset:
        print("Using subset for testing...")
        train_df, valid_df = train_test_split(processed_df, test_size=0.2, random_state=42)
        train_subset = train_df.sample(n=100, random_state=42)
    else:
        train_subset = processed_df

    print(f"Training records: {len(train_subset)}")

    # --------------------------
    # 4) Create ThePoet Pipeline
    # --------------------------
    print("Creating ThePoet pipeline...")
    try:
        poet_pipeline = create_thepoet_pipeline()
        print("ThePoet pipeline created.")
    except Exception as e:
        print(f"An error occurred while creating ThePoet pipeline: {e}")
        sys.exit(1)

    # --------------------------
    # 5) Train AraGPT2 Transformer Model
    # --------------------------
    transformer_name = "aubmindlab/aragpt2-base"
    try:
        transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        if transformer_tokenizer.pad_token is None:
            transformer_tokenizer.pad_token = transformer_tokenizer.eos_token
            print("Assigned EOS token as PAD token.")
    except Exception as e:
        print(f"Error loading tokenizer '{transformer_name}': {e}")
        sys.exit(1)
    try:
        transformer_model = AraGPT2ForClassicalStyle(model_name=transformer_name, freeze_layers=0, dropout_prob=0.1)
    except Exception as e:
        print(f"Error initializing AraGPT2 model: {e}")
        sys.exit(1)
    try:
        preprocessor = ArabertPreprocessor(model_name='aubmindlab/arabertv2')
        print("Initialized ArabertPreprocessor.")
    except Exception as e:
        print(f"Error initializing ArabertPreprocessor: {e}")
        sys.exit(1)
    try:
        trained_transformer, transformer_hist = train_aragpt2_for_classical_style(
            df_classical=train_subset,
            tokenizer=transformer_tokenizer,
            model=transformer_model,
            max_length=128,
            epochs=10,
            batch_size=4,
            output_dir=transformer_output_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            freeze_layers=0,
            weight_decay=0.01,
            patience=3,
            max_grad_norm=1.0
        )
        print("Transformer training complete.")
    except Exception as e:
        print(f"Error during AraGPT2 training: {e}")
        sys.exit(1)
    try:
        trained_transformer.save_pretrained(transformer_output_dir)
        transformer_tokenizer.save_pretrained(transformer_output_dir)
        print("Transformer model and tokenizer saved.")
    except Exception as e:
        print(f"Error saving Transformer model/tokenizer: {e}")
        sys.exit(1)
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(transformer_hist['train_loss'], label='Train Loss')
        plt.plot(transformer_hist['val_loss'], label='Validation Loss')
        plt.title("AraGPT2 Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error plotting training history: {e}")

    # --------------------------
    # 6) Train Diffusion Model with GPT2 Decoder
    # --------------------------
    print("Initializing Diffusion Model...")
    diffusion_model_params = {'num_transformer_blocks': 4, 'num_heads': 8, 'key_dim': 64, 'ffn_units': 512}
    input_shape = (max_bayt_len, encoding_dim)
    diffusion_model = create_diffusion_model_pytorch(input_shape, diffusion_model_params).to('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion_checkpoint_path = os.path.join(diffusion_output_dir, "diffusion_model_final.pt")
    if os.path.exists(diffusion_checkpoint_path):
        diffusion_model.load_state_dict(torch.load(diffusion_checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        print("Loaded pre-trained Diffusion Model.")
    else:
        print("No pre-trained Diffusion Model found. Using initialized model.")
    try:
        trained_diffusion, diffusion_hist = train_diffusion_with_gpt2_decoder(
            df_classical=train_subset,
            diffusion_model=diffusion_model,
            tokenizer=transformer_tokenizer,
            preprocessor=preprocessor,
            max_length=128,
            max_bayt_len=max_bayt_len,
            encoding_dim=encoding_dim,
            epochs=10,
            batch_size=8,
            output_dir=diffusion_output_dir,
            learning_rate=1e-4,
            patience=3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        final_diffusion_path = os.path.join(diffusion_output_dir, 'final_diffusion_model_with_decoder.pt')
        torch.save(trained_diffusion.state_dict(), final_diffusion_path)
        print(f"Final Diffusion model saved to '{final_diffusion_path}'")
    except Exception as e:
        print(f"Error during Diffusion Model training: {e}")
        sys.exit(1)

    # --------------------------
    # 7) Train Denoising Autoencoder (DAE)
    # --------------------------
    print("Training Denoising Autoencoder (DAE)...")
    texts_for_dae = train_subset['text'].tolist()
    try:
        dae_model, dae_hist = train_denoising_autoencoder(
            texts=texts_for_dae,
            tokenizer=transformer_tokenizer,
            max_length=128,
            epochs=5,
            batch_size=4,
            output_dir=dae_output_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"Error during DAE training: {e}")
        sys.exit(1)

    # --------------------------
    # 8) Train Masked Language Model (MLM)
    # --------------------------
    print("Training Masked Language Model (MLM)...")
    texts_for_mlm = train_subset['text'].tolist()
    try:
        mlm_model, mlm_tokenizer, mlm_hist = train_masked_language_model(
            texts=texts_for_mlm,
            model_name="aubmindlab/bert-base-arabertv2",
            max_length=128,
            epochs=3,
            batch_size=2,
            output_dir=mlm_output_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"Error during MLM training: {e}")
        sys.exit(1)

    # --------------------------
    # 9) Train Poetic Reinforcement Learning (RL) Model
    # --------------------------
    print("Training Poetic RL Model...")
    try:
        rl_model, rl_tokenizer = train_poetic_rl(
            model_name=transformer_name,
            initial_prompt="يا ليل الصب متى غده",
            episodes=5,
            max_length=20,
            alpha=0.9,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"Error during RL training: {e}")
        sys.exit(1)

    # --------------------------
    # 10) Generate Final Poem using the Full Model Chain
    # --------------------------
    modern_prompt = "يا جمال الزمان ويا نور الأمل"
    print("Generating final classical poem by chaining all models...")
    try:
        final_poem = generate_classical_poem_with_thepoet(
            modern_prompt=modern_prompt,
            poet_pipeline=poet_pipeline,
            transformer_model=trained_transformer,
            transformer_tokenizer=transformer_tokenizer,
            diffusion_model=trained_diffusion,
            dae_model=dae_model,
            mlm_model=mlm_model,
            max_length=128,
            max_bayt_len=max_bayt_len,
            encoding_dim=encoding_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("\n==== Final Chained Poem ====")
        print(final_poem)
        print("================================")
    except Exception as e:
        print(f"Error during poem generation: {e}")
