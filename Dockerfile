FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    tesseract-ocr-tur \
    tesseract-ocr-hin \
    tesseract-ocr-urd \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY streamlit-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit-app/ ./streamlit-app/

EXPOSE 10000

ENTRYPOINT ["sh", "-c", "streamlit run streamlit-app/main.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.maxUploadSize=500 --server.maxMessageSize=500"]
