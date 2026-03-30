FROM python:3.11-slim

WORKDIR /app

# Install dependencies
# --extra-index-url points to the PyTorch CPU-only wheel repository,
# which avoids downloading the much larger CUDA-enabled package (~2GB vs ~200MB).
COPY requirements.prod.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.prod.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Pre-download the default summarization model so the first request is fast.
# This caches the model weights inside the Docker image at build time.
# Change the model name here (and the SUMMARIZATION_MODEL env var) to switch models.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6'); \
    AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-6-6')"

# Copy application source
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
