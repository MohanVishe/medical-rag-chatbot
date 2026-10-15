import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.helper import download_hugging_face_embeddings, drop_reference_chunks, load_pdf, text_split

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "medical-bot")

extracted_data = load_pdf("data/")
if not extracted_data:
    raise SystemExit("No PDFs found in data/. See data/README.md.")
all_chunks = text_split(extracted_data)
text_chunks = drop_reference_chunks(all_chunks)
print(f"Dropped {len(all_chunks) - len(text_chunks)} reference-list chunks of {len(all_chunks)}.")
embeddings = download_hugging_face_embeddings()

# Start from an empty collection so re-running never duplicates chunks.
Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
       persist_directory=CHROMA_DIR).delete_collection()

# from_documents keeps each chunk's metadata (source file and page), so answers can cite pages.
Chroma.from_documents(
    text_chunks,
    embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)
print(f"Indexed {len(text_chunks)} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}.")
