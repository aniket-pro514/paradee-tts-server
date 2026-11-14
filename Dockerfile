FROM python:3.11-slim

ENV OMP_NUM_THREADS=1
ENV ONNXRUNTIME_INTRA_OP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    espeak-ng \
    espeak-ng-data \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm || true

COPY app.py .

# Pre-download and warm up Hindi, US English, and UK English models during Docker build!
RUN python -c "from app import load_paradee_engine, run_paradee_inference; load_paradee_engine(); run_paradee_inference('नमस्ते', 'hf_alpha', 1.0); run_paradee_inference('Hello world', 'af_heart', 1.0); run_paradee_inference('Good evening', 'bf_isabella', 1.0); print('All 6 voices pre-warmed!')" || true

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
