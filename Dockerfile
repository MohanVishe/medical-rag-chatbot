FROM python:3.10-slim

WORKDIR /app

# Build deps for sentence-transformers / ctransformers wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Build the Chroma index first (python store_index.py) or mount a pre-built one
# at /app/chroma_db before running this.
CMD ["python", "app.py"]
