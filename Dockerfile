FROM python:3.10-slim

WORKDIR /app

# Build deps for sentence-transformers / ctransformers wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# The model file and the Pinecone index must both exist before this runs.
# Mount the model: -v $(pwd)/model:/app/model
CMD ["python", "app.py"]
