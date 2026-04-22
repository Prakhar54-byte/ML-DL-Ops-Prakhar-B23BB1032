FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translate.py .

# Pre-download models and metrics
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-bn-en'); \
    AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-bn-en'); \
    import evaluate; evaluate.load('sacrebleu')"

CMD ["python", "translate.py"]