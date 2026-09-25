import os

from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
import pinecone

from src.helper import download_hugging_face_embeddings, load_pdf, text_split

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-bot")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

extracted_data = load_pdf("data/")
if not extracted_data:
    raise SystemExit("No PDFs found in data/. See data/README.md.")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")

# from_documents keeps each chunk's metadata (source file and page), so answers can cite pages.
Pinecone.from_documents(text_chunks, embeddings, index_name=INDEX_NAME)
print(f"Indexed {len(text_chunks)} chunks into '{INDEX_NAME}'.")
