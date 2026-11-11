FROM python:3.10-slim

WORKDIR /app

# The exact versions, with CPU-only torch (see requirements.txt for how the lock is made)
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

EXPOSE 8080

# Build the Chroma index first (python store_index.py) or mount a pre-built one
# at /app/chroma_db before running this.
CMD ["python", "app.py"]
