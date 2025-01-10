# Arabic Poetry Transformation Project

## Introduction

The Arabic Poetry Transformation Project aims to transform **modern Arabic poetry** into polished **Classical Arabic verse**. This is achieved through a hybrid approach that combines **Transformer-based language models** and **Diffusion models**. The goal is to generate poetry that adheres to the strict structural and stylistic rules of Classical Arabic poetry, including specific meters (علم العروض), consistent rhyme schemes, and elevated literary style.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Installing Dependencies](#installing-dependencies)
  - [Java Diacritization Setup](#java-diacritization-setup)
- [Running the Scripts](#running-the-scripts)
  - [Preprocessing Data](#preprocessing-data)
  - [Training the Transformer Model](#training-the-transformer-model)
  - [Training the Diffusion Model](#training-the-diffusion-model)
  - [Generating Output](#generating-output)
  - [Evaluating the Model](#evaluating-the-model)
- [Using the Notebooks](#using-the-notebooks)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

This project leverages advanced machine learning techniques to bridge the gap between modern and classical Arabic poetic forms.

**Objectives:**

1. **Preserve Semantic Meaning:** Ensure the transformed poem retains the essence of the original.
2. **Adhere to Classical Poetic Constraints:** Generate outputs that follow classical meters, rhyme schemes, and stylistic nuances.
3. **Provide Creative Refinement:** Introduce stylistic diversity while maintaining artistic quality.
4. **Demonstrate Model Superiority:** Showcase the effectiveness of the hybrid approach compared to general-purpose language models.

**Key Features:**

- **Hybrid AI Approach:**
  - **Transformers:** Handle linguistic and semantic understanding to generate initial drafts.
  - **Diffusion Models:** Iteratively refine outputs to enforce classical poetic constraints.
- **Evaluation Metrics:**
  - Adherence to classical meters and rhyme schemes.
  - Semantic preservation.
  - Human evaluation for artistic quality.

## Installation and Setup

### Prerequisites

- **Operating System:** Ubuntu 18.04 or later (Linux recommended)
- **Python Version:** Python 3.8 or higher
- **Java Development Kit (JDK):** Java 7 or Java 8 (required for diacritization scripts)
- **GPU:** NVIDIA GPU with CUDA support (recommended for training models)

### Setting Up the Environment

Choose **one** of the following methods to set up your Python environment:

#### Option 1: Using Python Virtual Environment

```bash
# Navigate to the project root directory
cd /path/to/your/project_root

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### Option 2: Using Conda

```bash
# Create a new conda environment
conda create -n arabic_poetry_env python=3.8

# Activate the environment
conda activate arabic_poetry_env
```

**Note:** Use either the Python virtual environment or Conda, not both, to avoid conflicts.

### Installing Dependencies

With the environment activated, install the required Python packages:

```bash
# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt
```

**Dependencies include:**

- numpy
- pandas
- tensorflow (compatible with your CUDA installation)
- keras
- scikit-learn
- pyarabic
- transformers
- torch
- matplotlib
- seaborn
- tensorflow-addons
- arabert
- arabert-preprocessor

### Java Diacritization Setup

**Required for diacritizing Arabic text using Farasa.**

#### Steps:

1. **Download Farasa Diacritizer:**

   - Register and download the Farasa Diacritizer JAR file from [Farasa's website](https://farasa.qcri.org/).
   - Place the `FarasaDiacritizeJar.jar` file and its dependencies in the `scripts/java/QCRI/Dev/ArabicNLP/Farasa/` directory.

2. **Set Up Java Environment:**

   - Ensure Java 7 or Java 8 is installed:

     ```bash
     java -version
     ```

     If not installed, install OpenJDK:

     ```bash
     sudo apt-get install openjdk-8-jdk
     ```

3. **Compile Java Programs:**

   Navigate to the `scripts/java` directory:

   ```bash
   cd scripts/java
   ```

   Compile the Java classes:

   ```bash
   javac -cp ".:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/FarasaDiacritizeJar.jar:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/lib/*" VerseDiacritizer.java
   ```

4. **Run Diacritization Example:**

   ```bash
   java -cp ".:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/FarasaDiacritizeJar.jar:./QCRI/Dev/ArabicNLP/Farasa/FarasaDiacritizeJar/dist/lib/*" VerseDiacritizer "أحب البرمجة بلغة الجافا"
   ```

   This should output the diacritized version of the input Arabic text.

**Note:** Ensure all paths are correctly set according to your directory structure.

## Running the Scripts

All scripts are located in the `scripts/python` directory. Ensure you are in the project root directory before running the scripts.

### Preprocessing Data

**Script:** `scripts/python/preprocess.py`

**Description:** Handles data preprocessing tasks for the Arabic Poetry Corpus Dataset (APCD).

**Usage:**

```bash
python scripts/python/preprocess.py \
  --input_file data/raw/apcd_full.csv \
  --output_dir data/processed \
  --tashkeel_threshold 0.4 \
  --allowed_eras "العصر الجاهلي" "العصر الأموي" \
  --allowed_meters "الطويل" "البسيط"
```

**Parameters:**

- `--input_file`: Path to the raw APCD CSV file.
- `--output_dir`: Directory to save processed data.
- `--tashkeel_threshold`: Minimum percentage of diacritics required.
- `--allowed_eras`: (Optional) List of eras to include.
- `--allowed_meters`: (Optional) List of meters to include.

### Training the Transformer Model

**Script:** `scripts/python/train_transformer.py`

**Description:** Trains a Transformer-based model (e.g., AraGPT2) to transform modern Arabic poetry into classical style.

**Usage:**

```bash
python scripts/python/train_transformer.py \
  --train_data data/processed \
  --model_name "aubmindlab/aragpt2-base" \
  --epochs 10 \
  --batch_size 4 \
  --output_dir models/transformers \
  --max_length 128
```

**Parameters:**

- `--train_data`: Directory containing processed `train.csv`, `valid.csv`, `test.csv`.
- `--model_name`: Name of the pre-trained transformer model to use.
- `--epochs`: Number of training epochs.
- `--batch_size`: Training batch size.
- `--output_dir`: Directory to save trained model checkpoints.
- `--max_length`: Maximum sequence length for tokenization.

### Training the Diffusion Model

**Script:** `scripts/python/train_diffusion.py`

**Description:** Trains a diffusion model to iteratively refine text embeddings.

**Usage:**

```bash
python scripts/python/train_diffusion.py \
  --train_data data/processed \
  --epochs 50 \
  --batch_size 32 \
  --output_dir models/diffusion \
  --model_params '{"num_transformer_blocks": 4, "num_heads": 8, "key_dim": 64, "ffn_units": 512}'
```

**Parameters:**

- `--train_data`: Directory containing processed `train.csv`, `valid.csv`, `test.csv`.
- `--epochs`: Number of training epochs.
- `--batch_size`: Training batch size.
- `--output_dir`: Directory to save trained model checkpoints.
- `--model_params`: JSON string of model parameters.

### Generating Output

**Script:** `scripts/python/generate_output.py`

**Description:** Uses the trained Transformer and Diffusion models to generate Classical Arabic poems from an input text.

**Usage:**

```bash
python scripts/python/generate_output.py \
  --input_file data/examples/modern_poems.txt \
  --output_dir results/outputs \
  --transformer_model_path models/transformers/transformer_model_final.pt \
  --transformer_model_name "aubmindlab/aragpt2-base" \
  --diffusion_model_path models/diffusion/diffusion_model_final.pt \
  --max_length 128
```

**Parameters:**

- `--input_file`: File containing modern Arabic poems (one per line).
- `--output_dir`: Directory to save generated classical poems.
- `--transformer_model_path`: Path to the trained transformer model.
- `--transformer_model_name`: Name of the transformer model used.
- `--diffusion_model_path`: Path to the trained diffusion model.
- `--max_length`: Maximum sequence length used during training.

### Evaluating the Model

**Script:** `scripts/python/evaluate.py`

**Description:** Evaluates model outputs against various metrics, such as meter accuracy, rhyme accuracy, and semantic preservation.

**Usage:**

```bash
python scripts/python/evaluate.py \
  --input_dir results/outputs \
  --metrics_output data/metrics \
  --diffusion_model_path models/diffusion/diffusion_model_final.pt \
  --transformer_model_path models/transformers/transformer_model_final.pt \
  --transformer_model_name "aubmindlab/aragpt2-base" \
  --max_length 128
```

**Parameters:**

- `--input_dir`: Directory containing generated poems to evaluate.
- `--metrics_output`: Directory to save evaluation results.
- `--diffusion_model_path`: Path to the trained diffusion model.
- `--transformer_model_path`: Path to the trained transformer model.
- `--transformer_model_name`: Name of the transformer model used.
- `--max_length`: Maximum sequence length used during training.

## Using the Notebooks

After installing all dependencies, you can explore and run the Jupyter notebooks located in the `notebooks/` directory. These notebooks provide an interactive environment for experimentation and visualization.

**Note:** Ensure that you have Jupyter Notebook or JupyterLab installed:

```bash
pip install jupyterlab
```

To start JupyterLab:

```bash
jupyter lab
```

## Project Structure

```
project_root/
├── data/
│   ├── raw/
│   │   └── apcd_full.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   └── test.csv
│   └── examples/
│       └── modern_poems.txt
├── models/
│   ├── transformers/
│   │   └── transformer_model_final.pt
│   └── diffusion/
│       └── diffusion_model_final.pt
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── outputs/
│   └── metrics/
├── scripts/
│   ├── python/
│   │   ├── preprocess.py
│   │   ├── train_transformer.py
│   │   ├── train_diffusion.py
│   │   ├── generate_output.py
│   │   ├── evaluate.py
│   │   └── utils.py
│   └── java/
│       ├── VerseDiacritizer.java
│       └── (other Java files and dependencies)
├── tests/
│   └── (unit tests)
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Acknowledgments:**

- Special thanks to [Hugging Face](https://huggingface.co/) for providing pre-trained Transformer models.
- Thanks to [Farasa](https://farasa.qcri.org/) for the diacritization tools.
- Inspired by the broader AI community for advancements in text generation models.
