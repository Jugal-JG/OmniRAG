FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MALLOC_ARENA_MAX=2
ENV HF_HOME=/data/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/data/.cache/sentence-transformers
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p uploads cache

EXPOSE 7860

CMD ["sh", "-c", "mkdir -p ${UPLOAD_FOLDER:-uploads} ${CACHE_FOLDER:-cache} ${HF_HOME:-/data/.cache/huggingface} ${SENTENCE_TRANSFORMERS_HOME:-/data/.cache/sentence-transformers} && gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 600 --access-logfile - --error-logfile -"]
