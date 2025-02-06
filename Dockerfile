# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies:
# - openjdk for running any Java-based tools (if needed)
# - git and build-essential for pip package compilation if required
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      openjdk-11-jre-headless \
      git \
      build-essential \
      ca-certificates \
      && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container.
# (Ensure that your saved model directories such as transformer_output are in the build context.)
COPY . .

# Install Python dependencies.
# If you have a requirements.txt file, it will be used. Otherwise, the default set is installed.
RUN if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir torch transformers arabert pyarabic h5py numpy pandas matplotlib scikit-learn tqdm; \
    fi

# (Optional) Pre-download the models that might be needed if they are not already saved.
# In this case, we assume you have already saved your models to disk.
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('aubmindlab/aragpt2-base')"

# Set the default command to run your inference script.
CMD ["python", "scripts/python/run_saved_models.py"]
